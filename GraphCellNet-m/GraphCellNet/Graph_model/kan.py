import torch
import torch.nn.functional as F
import math


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
     """
        Kolmogorov-Arnold Network (KAN) Linear layer implementation.
        
        This layer combines a traditional linear layer with B-spline interpolation
        to create a more expressive activation function that adapts to the data.
        
        Parameters
        ----------
        in_features : int
            Number of input features.
        out_features : int
            Number of output features.
        grid_size : int, optional
            Number of grid cells for the spline interpolation. Higher values allow
            for more complex activation functions. Default: 3.
        spline_order : int, optional
            Order of the B-spline. Higher values create smoother functions. Default: 3.
        scale_noise : float, optional
            Scale of the random noise used to initialize spline weights. Default: 0.1.
        scale_base : float, optional
            Scaling factor for the base linear component. Default: 1.0.
        scale_spline : float, optional
            Scaling factor for the spline component. Default: 1.0.
        enable_standalone_scale_spline : bool, optional
            Whether to use a learnable parameter for scaling spline weights. Default: True.
        base_activation : torch.nn.Module, optional
            Activation function to apply to the input before the base linear transformation.
            Default: torch.nn.SiLU.
        grid_eps : float, optional
            Blending factor between uniform and adaptive grid spacing. Default: 0.02.
        grid_range : list, optional
            Range of the grid [min, max]. Default: [-1, 1].
        
        Notes
        -----
        KANLinear layers provide a powerful alternative to traditional neural network
        layers by learning adaptable activation functions represented as B-splines.
        This allows the network to fit complex functions more efficiently.
        
        The layer combines a traditional linear transformation (with activation) and
        a spline-based transformation, where the spline coefficients are learned
        during training.
        """
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the layer's parameters.
        
        This method initializes:
        1. Base weights using Kaiming uniform initialization
        2. Spline weights using random noise and curve2coeff transformation
        3. Spline scalers (if enabled) using scaling factor or Kaiming uniform
        """
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions for the given input.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).
            
        Returns
        -------
        torch.Tensor
            B-spline basis functions of shape (batch_size, in_features, grid_size + spline_order).
            
        Notes
        -----
        This method implements the recursive definition of B-splines, computing 
        basis functions up to the specified spline order.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Convert a set of points (x, y) to B-spline coefficients.
        
        Parameters
        ----------
        x : torch.Tensor
            Input points tensor of shape (batch_size, in_features).
        y : torch.Tensor
            Output values tensor of shape (batch_size, in_features, out_features).
            
        Returns
        -------
        torch.Tensor
            B-spline coefficients of shape (out_features, in_features, grid_size + spline_order).
            
        Notes
        -----
        This method solves a least squares problem to find B-spline coefficients
        that best fit the given points.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        """
        Get the scaled spline weights.
        
        Returns
        -------
        torch.Tensor
            Spline weights scaled by the spline_scaler if enabled, otherwise original spline weights.
        """
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
         """
        Forward pass of the KANLinear layer.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., in_features).
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., out_features).
            
        Notes
        -----
        The forward pass combines:
        1. A traditional linear transformation with activation function
        2. A spline-based transformation using learned B-spline coefficients
        """
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        """
        Update the grid based on the data distribution.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).
        margin : float, optional
            Margin to add around the data range. Default: 0.01.
            
        Notes
        -----
        This method:
        1. Computes the current output of the spline component
        2. Updates the grid to better match the data distribution
        3. Updates the spline weights to maintain the same function
        
        This adaptation allows the network to focus more resolution where the
        data is concentrated, improving approximation quality.
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute regularization loss for the spline weights.
        
        Parameters
        ----------
        regularize_activation : float, optional
            Weight for the activation regularization term (L1 norm). Default: 1.0.
        regularize_entropy : float, optional
            Weight for the entropy regularization term. Default: 1.0.
            
        Returns
        -------
        torch.Tensor
            Combined regularization loss.
            
        Notes
        -----
        This method computes two types of regularization:
        1. Activation regularization: encourages sparse spline weights
        2. Entropy regularization: encourages uniform distribution of weights
        
        These regularizations help produce simpler and more robust functions.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
    """
        Kolmogorov-Arnold Network (KAN) implementation.
        
        A KAN is a neural network architecture that uses KANLinear layers to approximate
        functions using adaptive B-spline activation functions.
        
        Parameters
        ----------
        layers_hidden : list
            List of integers defining the network architecture, where each integer
            represents the number of neurons in that layer.
        grid_size : int, optional
            Number of grid cells for the spline interpolation. Default: 5.
        spline_order : int, optional
            Order of the B-spline. Default: 3.
        scale_noise : float, optional
            Scale of the random noise used to initialize spline weights. Default: 0.1.
        scale_base : float, optional
            Scaling factor for the base linear component. Default: 1.0.
        scale_spline : float, optional
            Scaling factor for the spline component. Default: 1.0.
        base_activation : torch.nn.Module, optional
            Activation function to apply to the input before the base linear transformation.
            Default: torch.nn.SiLU.
        grid_eps : float, optional
            Blending factor between uniform and adaptive grid spacing. Default: 0.02.
        grid_range : list, optional
            Range of the grid [min, max]. Default: [-1, 1].
            
        Notes
        -----
        KANs are a modern architecture that combines the universal approximation capabilities
        of neural networks with the interpretability of B-splines. They can achieve high 
        performance with fewer parameters than traditional neural networks for many tasks.
        
        The architecture was inspired by the Kolmogorov-Arnold representation theorem,
        which states that any multivariate continuous function can be represented as a 
        superposition of continuous functions of one variable.
        """
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
         """
        Forward pass of the KAN network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features).
        update_grid : bool, optional
            Whether to update the grids of each layer based on the current batch.
            This adaptation can improve performance but should typically only be
            done during initial training. Default: False.
            
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_features).
            
        Notes
        -----
        If update_grid is True, each layer's grid will be updated based on the
        current data distribution at that layer. This can help the network
        allocate more resolution to areas of the input space that are more
        relevant to the task.
        """
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the combined regularization loss across all layers.
        
        Parameters
        ----------
        regularize_activation : float, optional
            Weight for the activation regularization term (L1 norm). Default: 1.0.
        regularize_entropy : float, optional
            Weight for the entropy regularization term. Default: 1.0.
            
        Returns
        -------
        torch.Tensor
            Combined regularization loss from all layers.
            
        Notes
        -----
        This method aggregates the regularization losses from all KANLinear layers
        in the network. It helps control the complexity of the learned functions
        and prevent overfitting.
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
