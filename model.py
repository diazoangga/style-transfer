import tensorflow as tf

def build_model(model_arch='vgg16', weights='imagenet'):
    if model_arch == 'inceptionV3':
        print('\nBuilding model using pretrained model: {}...'.format(model_arch))
        pretrained_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights=weights)
        # Content layer where will pull our feature maps
        content_layers = ['block5_conv2'] 
        # Style layer we are interested in
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1',
                        'block5_conv1'
                    ]
        
    elif model_arch == 'Xception':
        print('\nBuilding model using pretrained model: {}...'.format(model_arch))
        pretrained_model = tf.keras.applications.xception.Xception(include_top=False, weights=weights)
        # Content layer where will pull our feature maps
        content_layers = ['block5_conv2'] 
        # Style layer we are interested in
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1',
                        'block5_conv1'
                    ]
        
    elif model_arch == 'vgg16':
        print('\nBuilding model using pretrained model: {}...'.format(model_arch))
        pretrained_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights=weights)
        # Content layer where will pull our feature maps
        content_layers = ['block5_conv2'] 
        # Style layer we are interested in
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1',
                        'block5_conv1'
                    ]
    else:
        print('\n{} doesnt seem the right naming for the pretrained model.\nBuilding model using pretrained model: VGG16...'.format(model_arch))
        pretrained_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights=weights)
        # Content layer where will pull our feature maps
        content_layers = ['block5_conv2'] 
        # Style layer we are interested in
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1',
                        'block5_conv1'
                    ]

    pretrained_model.trainable = False
    layer_names = content_layers + style_layers
    outputs = [pretrained_model.get_layer(name).output for name in layer_names]
    
    model = tf.keras.Model(inputs=pretrained_model.input, outputs=outputs)
    
    return model