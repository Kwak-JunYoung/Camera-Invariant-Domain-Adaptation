from models.unet import UNet

# Obtain information of the model
def get_model_info(config, model_name):
    if model_name == "unet":
        model_config = config["models"]["unet"]
        model = UNet()
    else: 
        raise NotImplementedError("model name is not valid")
    return model_config, model

