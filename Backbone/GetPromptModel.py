"""
build_promptmodel   Script  ver： Oct 17th 14:20

"""

try:
    from Backbone.VPT_structure import *
except:
    from Backbone.VPT_structure import *


def build_promptmodel(num_classes=1000, edge_size=224, model_idx='ViT', patch_size=16,
                      Prompt_Token_num=20, VPT_type="Deep", prompt_state_dict=None, base_state_dict='timm'):
    """
    following the https://github.com/sagizty/VPT
    this build the VPT (prompt version of ViT), with additional prompt tokens,
    each layer the information become [B, N_patch + N_prompt, Dim]

    During training only the prompt tokens and the head layer are
    set to be learnable while the rest of Transformer layers are frozen

    # VPT_type = "Shallow" / "Deep"
        - Shallow: concatenate N_prompt of prompt tokens before the first Transformer Encoder block,
                each layer the information become [B, N_patch + N_prompt, Dim]
        - Deep: concatenate N_prompt of prompt tokens to each Transformer Encoder block,
                this will replace the output prompt tokens learnt form previous encoder.
    """

    if model_idx[0:3] == 'ViT':

        if base_state_dict is None:
            basic_state_dict = None

        elif type(base_state_dict) == str:
            if base_state_dict == 'timm':
                # ViT_Prompt
                import timm
                # from pprint import pprint
                # model_names = timm.list_models('*vit*')
                # pprint(model_names)

                basic_model = timm.create_model('vit_base_patch' + str(patch_size) + '_' + str(edge_size),
                                                pretrained=True)
                basic_state_dict = basic_model.state_dict()
                print('in prompt model building, timm ViT loaded for base_state_dict')

            else:
                basic_state_dict = None
                print('in prompt model building, no vaild str for base_state_dict')

        else:  # state dict: collections.OrderedDict
            basic_state_dict = base_state_dict
            print('in prompt model building, a .pth base_state_dict loaded')

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type, basic_state_dict=basic_state_dict)

        model.New_CLS_head(num_classes)

        if prompt_state_dict is not None:
            try:
                model.load_prompt(prompt_state_dict)
            except:
                print('erro in .pth prompt_state_dict')
            else:
                print('in prompt model building, a .pth prompt_state_dict loaded')

        model.Freeze()
    else:
        print("The model is not difined in the Prompt script！！")
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        print('Build VPT model with in/out shape: ', img.shape, ' -> ', preds.shape)

    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model


if __name__ == '__main__':
    model = build_promptmodel(prompt_state_dict=None, base_state_dict='timm', num_classes=0)
