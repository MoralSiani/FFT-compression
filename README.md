# FFT-compression
Proof of concept for audio and image compression using a custom implementation of the FFT algorithm.

This program compresses and decompresses `*.wav` or `*.bmp` files by truncating higher frequencies.
Additionally, the program can generate visual representations of the time/color and frrequency domain. 

![Analysis output](/compression.png)

```
usage: main.py FILE [options]
                                                                                                    
positional arguments:                                                                               
  FILE                  Input file (.wav or .bmp)                                                   
  
options:                                                                                            
  -h, --help            Show this help message and exit                                             
  -a, --analyze         Compress, decompress and plot graphs (only bmp and wav files)     
  -o OUTPUT_DIR         Output directory [default: data]
  -c, --clear           Clear output directory
```
