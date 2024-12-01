from transformers import GPT2LMHeadModel
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')
from matplotlib.backends.backend_webagg import FigureCanvasWebAgg
import tornado.ioloop
import tornado.web

# Set the desired port for WebAgg
desired_port = 8081  # The port you want to use for the WebAgg server

# Load the GPT-2 model
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")  # 124M
sd_hf = model_hf.state_dict()

# Watch the layers, especially token embeddings and position embeddings
for k, v in sd_hf.items():
    print(k, v.shape)

# Print some weights
print(sd_hf['transformer.wte.weight'].view(-1)[:20])

# Plot the weights
fig = plt.figure()
plt.imshow(sd_hf['transformer.wte.weight'], cmap='gray')

# Adjust plot limits
plt.xlim(0, 768)  # Adjust based on your data
plt.ylim(0, 1024)  # Adjust based on your data
plt.gca().set_aspect('auto', adjustable='box')

# Create a canvas for the figure
canvas = FigureCanvasWebAgg(fig)

# Define a simple Tornado handler for the WebAgg application
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "image/png")
        canvas.print_png(self)  # Sends the plot as a PNG image

# Create the Tornado application and pass the handler
application = tornado.web.Application([
    (r"/", MainHandler),
])

# Start the Tornado server on the desired port
if __name__ == "__main__":
    print(f"WebAgg server running on http://localhost:{desired_port}")
    application.listen(desired_port)  # Ensure the server listens on the correct port
    tornado.ioloop.IOLoop.current().start()
