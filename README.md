# LCM_Inpaint-Outpaint_Comfy
ComfyUI custom nodes for inpainting/outpainting using the new latent consistency model (LCM)

# Inpaint
![Screenshot from 2023-10-24 22-38-53](https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy/assets/57796911/c81da663-c71e-4035-a7d7-e29294315054)


# Outpaint
![Uploading Screenshot from 2023-10-24 22-42-53.png…]()



# How to Use:
Clone into custom_nodes folder inside your ComfyUI directory
   ```
   git clone https://github.com/taabata/LCM_Inpaint-Outpaint_Comfy
   ```

Download the model in diffusers format from https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/tree/main and place it inside model/diffusers folder in your ComfyUI directory. (The name of the model folder should be "LCM_Dreamshaper_v7")

Load the workflow by choosing the .json file for inpainting or outpainting.



# Credits
Done by refering to nagolinc's img2img script and the diffuser's inpaint pipeline

