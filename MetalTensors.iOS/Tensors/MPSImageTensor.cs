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
            : this (new MPSImage (device, MPSImageDescriptor.GetImageDescriptor (
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

            fixed (float* dest = destination) {
                image.ReadBytes ((IntPtr)dest, dataLayout, 0);
            }

            throw new NotImplementedException ();
        }
    }
}
