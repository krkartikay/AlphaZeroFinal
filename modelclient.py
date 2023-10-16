import requests
import config
import torch

class Model:
    def __init__(self):
        self.server_url = config.server_address
    
    def predict_image(self, image):
        response = requests.get(f"{self.server_url}/predict", json=image.tolist())
        
        if response.status_code == 200:
            return response.json().get("prediction")
        else:
            raise Exception("Error occurred during prediction: {}".format(response.json().get("error")))

    def predict(self, gamestate):
        probs, value = self.predict_image(gamestate.to_image())
        return torch.Tensor(probs), torch.Tensor(value)

# Example Usage:
if __name__ == "__main__":
    model = Model()

    import game
    g = game.GameState()
    print(model.predict(g))
