import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3Plus(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet3Plus, self).__init__()
        filters = [64, 128, 256, 512, 1024]

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(5):
            if i == 0:
                in_channels = input_channels
            else:
                in_channels = filters[i - 1]
            self.encoders.append(self.conv_block(in_channels, filters[i]))

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(5):
            in_channels = filters[0]
            out_channels = filters[i]
            self.decoders.append(self.upconv_block(in_channels, out_channels))

        # Final convolution layer
        self.final_conv = nn.Conv2d(filters[0], output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        encoder_outputs = []
        
        # Encoder pass
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)

        # Decoder pass
        for i, decoder in enumerate(self.decoders):
            encoder_output = encoder_outputs[-(i + 2)]  # Skip connection from encoder
            x = decoder(x)
            x = torch.cat([x, encoder_output], dim=1)  # Concatenate skip connection
        
        # Final convolution
        x = self.final_conv(x)
        x = F.softmax(x, dim=1)

        return x

if __name__ == "__main__":
    INPUT_CHANNELS = 1
    OUTPUT_CHANNELS = 1

    model = UNet3Plus(INPUT_CHANNELS, OUTPUT_CHANNELS)
    print(model)
