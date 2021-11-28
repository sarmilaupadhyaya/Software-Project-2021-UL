This Dockerfile creates an image of the Grad-TTS system that is compatible that can be run on a GPU node.
For Grid-5000 users, first install Docker on the node "g5k-setup-nvidia-docker -t".
Then build the image with "docker build ."
Run a container with docker run -it [container] /bin/bash
Download/transfer the model checkpoints to the container.
