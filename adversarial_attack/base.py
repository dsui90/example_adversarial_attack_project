
class WhiteBoxBaseClass():
    """
    Base class for white-box adversarial attacks.
    """

    def __init__(self, model):
        """
        Initialize the attack with the model and any other parameters.

        Args:
            model: The model to attack.
        """
        self.model = model

    def generate(self, source_img, gt_label=None, target_label=None, **kwargs):
        """
        Generate adversarial examples.

        Args:
            source_img: Input data.
            gt_label: Ground truth labels (optional).
            target_label: Target labels (optional).
            **kwargs: Additional parameters for the attack.

        Returns:
            Adversarial examples.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class BlackBoxBaseClass():
    """
    Base class for black-box adversarial attacks.
    """

    def generate(self, source_img, gt_label=None, target_label=None, **kwargs):
        """
        Generate adversarial examples.

        Args:
            source_img: Input data.
            gt_label: Ground truth labels (optional).
            target_label: Target labels (optional).
            **kwargs: Additional parameters for the attack.

        Returns:
            Adversarial examples.
        """
        raise NotImplementedError("Subclasses should implement this method.")