import torch
import torch.nn as nn
import copy

class RRAMNonidealities:
    """
    A class to apply various nonidealities commonly encountered in RRAM-based
    analog in-memory computing systems. This includes:

      1) Quantization: finite conductance levels (num_levels).
      2) Device-to-device (D2D) variation.
      3) Cycle-to-cycle (C2C) variation.
      4) Read noise.

    Attributes:
    -----------
    d2d_std : float
        Standard deviation for device-to-device variation (percentage factor).
    c2c_std : float
        Standard deviation for cycle-to-cycle variation (percentage factor).
    read_noise_std : float
        Standard deviation for read noise (percentage factor).
    num_levels : int
        Number of quantization levels (e.g., 256 -> 8-bit).

    Example Usage:
    --------------
    >>> rram_noise = RRAMNonidealities(d2d_std=0.01, c2c_std=0.005,
                                       read_noise_std=0.002, num_levels=256)
    >>> # Apply each non-ideality independently
    >>> quantized_model = rram_noise.apply_quantization(pretrained_model)
    >>> d2d_model = rram_noise.apply_device_to_device_variation(pretrained_model)
    >>> c2c_model = rram_noise.apply_cycle_to_cycle_variation(pretrained_model)
    >>> read_noise_model = rram_noise.apply_read_noise(pretrained_model)
    >>>
    >>> # Or apply all non-idealities in sequence
    >>> noisy_model_all = rram_noise.apply_all_nonidealities(pretrained_model)
    """

    def __init__(
        self,
        d2d_std=0.01,   # Device-to-device std dev (percentage)
        c2c_std=0.005,  # Cycle-to-cycle std dev (percentage)
        read_noise_std=0.002,  # Additional read noise (percentage)
        num_levels=256  # Quantization levels
    ):
        self.d2d_std = d2d_std
        self.c2c_std = c2c_std
        self.read_noise_std = read_noise_std
        self.num_levels = num_levels

    ###########################################################################
    # Internal "private" methods for transformations on a single tensor
    ###########################################################################

    def _apply_quantization(self, w: torch.Tensor) -> torch.Tensor:
        """
        Internal method: Applies quantization to simulate finite RRAM conductance levels.
        """
        if self.num_levels > 0:
            return quantize_values(w, self.num_levels)
        return w

    def _apply_d2d_variation(self, w: torch.Tensor) -> torch.Tensor:
        """
        Internal method: Applies device-to-device variation (additive noise
        proportional to the absolute value of each weight).
        """
        if self.d2d_std > 0:
            d2d_noise = torch.randn_like(w) * (self.d2d_std * w.abs())
            w = w + d2d_noise
        return w

    def _apply_c2c_variation(self, w: torch.Tensor) -> torch.Tensor:
        """
        Internal method: Applies cycle-to-cycle variation (another additive noise
        proportional to the absolute value of each weight).
        """
        if self.c2c_std > 0:
            c2c_noise = torch.randn_like(w) * (self.c2c_std * w.abs())
            w = w + c2c_noise
        return w

    def _apply_read_noise(self, w: torch.Tensor) -> torch.Tensor:
        """
        Internal method: Applies read noise (final read-stage noise proportional to
        the absolute value of each weight).
        """
        if self.read_noise_std > 0:
            read_noise = torch.randn_like(w) * (self.read_noise_std * w.abs())
            w = w + read_noise
        return w

    ###########################################################################
    # Public methods that apply each non-ideality independently
    ###########################################################################

    def apply_quantization(self, model: nn.Module) -> nn.Module:
        """
        Applies only quantization to the model's parameters.

        Returns
        -------
        nn.Module
            A new model instance with quantized weights/biases.
        """
        noisy_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in noisy_model.named_parameters():
                if "weight" in name or "bias" in name:
                    w = param.data
                    w = self._apply_quantization(w)
                    param.data = w
        return noisy_model

    def apply_device_to_device_variation(self, model: nn.Module) -> nn.Module:
        """
        Applies only device-to-device (D2D) variation to the model's parameters.

        Returns
        -------
        nn.Module
            A new model instance with D2D variation applied.
        """
        noisy_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in noisy_model.named_parameters():
                if "weight" in name or "bias" in name:
                    w = param.data
                    w = self._apply_d2d_variation(w)
                    param.data = w
        return noisy_model

    def apply_cycle_to_cycle_variation(self, model: nn.Module) -> nn.Module:
        """
        Applies only cycle-to-cycle (C2C) variation to the model's parameters.

        Returns
        -------
        nn.Module
            A new model instance with C2C variation applied.
        """
        noisy_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in noisy_model.named_parameters():
                if "weight" in name or "bias" in name:
                    w = param.data
                    w = self._apply_c2c_variation(w)
                    param.data = w
        return noisy_model

    def apply_read_noise(self, model: nn.Module) -> nn.Module:
        """
        Applies only read noise to the model's parameters.

        Returns
        -------
        nn.Module
            A new model instance with read noise applied.
        """
        noisy_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in noisy_model.named_parameters():
                if "weight" in name or "bias" in name:
                    w = param.data
                    w = self._apply_read_noise(w)
                    param.data = w
        return noisy_model

    ###########################################################################
    # Public method to apply ALL non-idealities in one go
    ###########################################################################

    def apply_all_nonidealities(self, model: nn.Module) -> nn.Module:
        """
        Inject approximate RRAM crossbar nonidealities into model parameters,
        applying all of the following in sequence:
          1) Quantization
          2) Device-to-device variation
          3) Cycle-to-cycle variation
          4) Read noise

        Returns
        -------
        nn.Module
            A new model instance with all nonidealities applied in sequence.
        """
        noisy_model = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in noisy_model.named_parameters():
                if "weight" in name or "bias" in name:
                    w = param.data
                    w = self._apply_quantization(w)
                    w = self._apply_d2d_variation(w)
                    w = self._apply_c2c_variation(w)
                    w = self._apply_read_noise(w)
                    param.data = w
        return noisy_model