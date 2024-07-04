## Docker Container

The original installation instructions are laid out for a Unix-like OS, for example the Linux-based Ubuntu OS. 
If you don't have a Linux OS and would like to install this repo on a Windows computer, a solution is to use a Docker container. 
A possible docker container installation could look as follows: 

### Docker Desktop Installation and Setup under Windows

1. **Install Docker Desktop** from the [official Docker website](https://www.docker.com/products/docker-desktop/) and launch the application.

2. **Verify Docker installation** by running the following command in a Command Prompt, Terminal, or PowerShell:
    ```sh
    docker --version
    ```

3. **Install VcXsrv**: In order to be display graphical applications running on a Linus operating system on a windows computer, you can for example install 'VcXsrv', which is an open-source Windows X Server.
                       VcXsrv can be downloaded from the [official SourceForge page](https://sourceforge.net/projects/vcxsrv/)
   
     ***Start VcXsrv*** and use your preferred display settings, for example the default options 'Multiple Windows' and 'Display Number':-1. Then click 'Next' and make sure 'Start no client' is selected. After another click on 'Next' you get to the extra settings which you can leave at the default values. Click on "Finish" to start the X Server.
        
   
5. **Get the Dockerfile**: Download the Dockerfile directly from this folder (../Dokcer/Dockerfile) or use the Dockerfile in your cloned repository and place it in the desired directory from where you want to manage your Docker container.

6. **Place the Dockerfile into your desired project directory**:
    ```sh
    cd your\project\directory
    ```

7. **Within your project directory, build your Docker image via the `docker build` command**:
    ```sh
    docker build -t <your-docker-image-name> .
    ```
    Replace `<your-docker-image-name>` with a suitable name, for example, `nvidia_cuda_118`. 

8. **Create your docker container** based on the newly created docker image `<your-docker-image-name>` via the 'docker run' command: In order to be able to view graphical applications running within your Linux-based docker container on a Windows computer you have to use the '--env="DISPLAY' option.
   If you want to use available GPUs you can activat this option via the '--gpus' argument, wither select specific GPUs via their number (0, 1, 2, etc.) or use all of them ('--gpus all'): 
    ```sh
   docker run --name `<your-docker-container-name>` --gpus all --net=host --env="DISPLAY" -it `<your-docker-image-name>`
   ```

9. **Check status of your Docker container**: Running Docker images and containers can be managed via the Docker Desktop application, which is mostly self-explanatory when using Docker Desktop's graphical interface. You should now verify that the newly created Docker container is running.
    If you prefer working with a command line prompt, use the following commands to 1.) check that your container is running:
     ```sh
      docker ps
    ```
    ... and enter your newly created docker container:
    ```sh
    docker exec <your-docker-image-name> /bin/bash
    ```
11. In order to use all benefits of a code editor like Visual Studio Code (VS Code) when working with GitGub repos and Python coding within a running Docker container, the following can be done:
     * Install Visual Studio Code from [the official Visual Studio Code website](https://code.visualstudio.com/)
     * Connect to your running Docker container:
       - Install the 'Remote - Containers' extention in VS Code
       - Open the Coomman Palette in VS Code (press 'Ctrl+Shift+P')
       - Type 'Remote-Containers: Attach to Running Container.." and select it
       - Now a list of all running containers appears. Select your newly created container in order to connect to it.
       - VS Code will then open in a new window which is connected to your running Docker container
       - Now the filesystem of your container is available from your VS Code explorer, which allows you to develop code, open terminals, run commands and launch graphical applications from termainals (since you have VcXsrv installed and running). 
