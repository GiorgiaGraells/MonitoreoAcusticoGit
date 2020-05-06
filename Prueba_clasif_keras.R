

## importing

library(stringr)
library(dplyr)

files <- list.files(path = "data", pattern = ".wav", recursive = T, full.names = T)

files <- files[!str_detect(files, "background_noise")]

df <- data.frame(fname = files) %>% mutate(class = str_extract(fname ,"1/.*/"))

df$class <- df$fname %>% str_split("/", simplify = T) %>% as.data.frame() %>% pull(V2) %>% as.character()

df <- df %>% mutate(class_id = class %>% as.factor() %>% as.integer() - 1L)

#gnerating

library(tfdatasets)
ds <- tensor_slices_dataset(df)

window_size_ms <- 30
window_stride_ms <- 10 

window_size <- as.integer(16000*window_size_ms/1000)
stride <- as.integer(16000*window_stride_ms/1000) #revisar propiedades de audios 

fft_size <- as.integer(2^trunc(log(window_size, 2)) + 1)
n_chunks <- length(seq(window_size/2, 16000 - window_size/2, stride))


##################### donde dejamos el miercoles 6 mayo con derek

# shortcuts to used TensorFlow modules.
audio_ops <- tf$contrib$framework$python$ops$audio_ops

ds <- ds %>%
  dataset_map(function(obs) {
    
    # a good way to debug when building tfdatsets pipelines is to use a print
    # statement like this:
    # print(str(obs))
    
    # decoding wav files
    audio_binary <- tf$read_file(tf$reshape(obs$fname, shape = list()))
    wav <- audio_ops$decode_wav(audio_binary, desired_channels = 1)
    
    # create the spectrogram
    spectrogram <- audio_ops$audio_spectrogram(
      wav$audio, 
      window_size = window_size, 
      stride = stride,
      magnitude_squared = TRUE
    )
    
    # normalization
    spectrogram <- tf$log(tf$abs(spectrogram) + 0.01)
    
    # moving channels to last dim
    spectrogram <- tf$transpose(spectrogram, perm = c(1L, 2L, 0L))
    
    # transform the class_id into a one-hot encoded vector
    response <- tf$one_hot(obs$class_id, 30L)
    
    list(spectrogram, response)
  })
