import tensorflow as tf

def gram_matrix(input_tensor):
#     input_tensorT = tf.transpose(input_tensor)
    result = tf.linalg.einsum('bijc, bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/num_locations

def extract_style_content(img_tensor, model) :
    preprocessed_img = tf.keras.applications.vgg19.preprocess_input(img_tensor*255.0) # preprocessing specific to vgg19 model. Multiplying by 255. since preprocess function expects float inputs in range 0-255. 
    outputs = model(preprocessed_img)
    content_output = outputs[-1] # selecting last output and 
    style_outputs  = outputs[:-1] # a list of style layer outputs 
    style_outputs  = [gram_matrix(style_output) for style_output in style_outputs] # extracting style for each style layer using Gram matrix
    return style_outputs, content_output

def custom_loss(style_outputs, content_output, target_style_outputs, target_content_output, style_weight=1e-2, content_weight=1e-32) :

    style_layer_losses = [ tf.reduce_mean((output - target_output)**2) for output, target_output in zip(style_outputs, target_style_outputs)  ]
    style_loss = tf.add_n(style_layer_losses)/len(style_layer_losses)

    content_loss = tf.reduce_mean( (content_output - target_content_output)**2  )

    total_loss = style_weight*style_loss + content_weight*content_loss
    return total_loss, content_loss, style_loss