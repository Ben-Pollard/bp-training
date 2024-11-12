from transformers import AutoConfig, AutoModelForTokenClassification

class TokenClassificationModel:
    def __init__(self, model_name_or_path: str, num_labels: int):
        """
        Initializes the TokenClassificationModel with the given parameters.

        Args:
            model_name_or_path (str): Path to the pre-trained model or model identifier.
            num_labels (int): Number of labels for classification.
        """
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, config=self.config
        )

    def get_model(self):
        """
        Returns the model instance.

        Returns:
            AutoModelForTokenClassification: The model instance.
        """
        return self.model
