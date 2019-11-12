using System;
using CoreGraphics;
using Foundation;
using Metal;
using MetalKit;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class MPSImageTensor : Tensor
    {
        readonly int[] shape;
        readonly MPSImage image;

        public override int[] Shape => shape;

        public MPSImage Image => image;

        public MPSImageTensor (MPSImage image)
        {
            this.image = image;
            shape = new[] { (int)image.Height, (int)image.Width, (int)image.FeatureChannels };
        }

        public MPSImageTensor (IMTLTexture texture, int featureChannels = 3)
            : this (new MPSImage (texture, (nuint)featureChannels))
        {
        }

        public MPSImageTensor (MPSImageDescriptor descriptor, IMTLDevice? device = null)
            : this (new MPSImage (device.Current (), descriptor))
        {
        }

        public MPSImageTensor (int height, int width, int featureChannels = 3, IMTLDevice? device = null)
            : this (new MPSImage (device.Current (), MPSImageDescriptor.GetImageDescriptor (
                MPSImageFeatureChannelFormat.Float32, (nuint)width, (nuint)height, (nuint)featureChannels)))
        {
        }

        public MPSImageTensor (NSUrl url, int featureChannels = 3, IMTLDevice? device = null)
        {
            if (url is null) {
                throw new ArgumentNullException (nameof (url));
            }

            var dev = device.Current ();
            using var loader = new MTKTextureLoader (dev);
            var texture = loader.FromUrl (url, null, out var error);
            error.ValidateNoError ();

            image = new MPSImage (texture, (nuint)featureChannels);
            shape = new[] { (int)image.Height, (int)image.Width, (int)image.FeatureChannels };
        }

        public MPSImageTensor (string path, int featureChannels = 3, IMTLDevice? device = null)
            : this (NSUrl.FromFilename (path), featureChannels, device)
        {
        }

        public MPSImageTensor (CGImage cgimage, int featureChannels = 3, IMTLDevice? device = null)
        {
            if (cgimage is null) {
                throw new ArgumentNullException (nameof (cgimage));
            }

            var dev = device.Current ();
            using var loader = new MTKTextureLoader (dev);
            var texture = loader.FromCGImage (cgimage, null, out var error);
            error.ValidateNoError ();

            image = new MPSImage (texture, (nuint)featureChannels);
            shape = new[] { (int)image.Height, (int)image.Width, (int)image.FeatureChannels };
        }

        public unsafe override void Copy (Span<float> destination)
        {
            ValidateCopyDestination (destination);
            var dataLayout = MPSDataLayout.HeightPerWidthPerFeatureChannels;
            var dtype = image.PixelFormat;
            switch (dtype) {
                case MTLPixelFormat.R32Float: {
                        fixed (float* dataPtr = destination) {
                            image.ReadBytes ((IntPtr)dataPtr, dataLayout, 0);
                        }
                    }
                    break;
                case MTLPixelFormat.RGBA32Float: {
                        fixed (float* dataPtr = destination) {
                            image.ReadBytes ((IntPtr)dataPtr, dataLayout, 0);
                        }
                    }
                    break;
                default:
                    throw new NotSupportedException ($"Cannot copy image with pixel format {dtype}");
            }
        }

        public unsafe override Tensor Slice (params int[] indexes)
        {
            var pixelFormat = image.PixelFormat;
            //Console.WriteLine ("Pixel Format = " + pixelFormat);

            if (indexes.Length >= 2) {
                // Single pixel or channel
                var y = (nuint)indexes[0];
                var x = (nuint)indexes[1];
                var numChannels = (int)image.FeatureChannels;
                var numImages = (numChannels + 3) / 4;
                var region = MTLRegion.Create3D (x, y, 0, 1, 1, (nuint)numImages);
                var imageIndex = 0;
                var featureChannelInfo = new MPSImageReadWriteParams {
                    NumberOfFeatureChannelsToReadWrite = (nuint)numChannels
                };
                var dataLayout = MPSDataLayout.HeightPerWidthPerFeatureChannels;

                switch (pixelFormat) {
                    case MTLPixelFormat.BGRA8Unorm_sRGB when numChannels == 3: {
                            var dtypeSize = sizeof (byte);
                            var bytesPerRow = (nuint)(numChannels * dtypeSize);
                            var bytesPerImage = bytesPerRow;
                            var dataPtr = stackalloc byte[numChannels];
                            //var rawData = new byte[(int)(numImages * bytesPerImage)];
                            //fixed (byte* dataPtr = rawData) {
                                image.ReadBytes ((IntPtr)dataPtr, dataLayout, bytesPerRow, bytesPerImage, region, featureChannelInfo, (nuint)imageIndex);
                            //}
                            var floatScale = 1.0f / 255.0f;
                            if (indexes.Length == 2) {
                                var floatData = new float[numChannels];
                                for (var i = 0; i < numChannels; i++) {
                                    floatData[i] = dataPtr[2 - i] * floatScale;
                                }
                                return Tensor.Array (floatData);
                            }
                            return Tensor.Array (new float[] { dataPtr[2 - indexes[2]] * floatScale });
                        }
                }
            }
            return base.Slice (indexes);
        }

        public override MPSImage GetImage ()
        {
            return image;
        }
    }
}
