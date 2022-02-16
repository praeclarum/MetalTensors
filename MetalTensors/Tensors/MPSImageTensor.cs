﻿using System;
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

        public MPSImage MetalImage => image;
        public IMTLDevice? Device => image.Device;

        public override bool IsStatic => true;

        public MPSImageTensor (MPSImage image)
        {
            if (image == null || image.Handle == IntPtr.Zero)
                throw new ArgumentNullException (nameof(image));
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

        public MPSImageTensor (nuint height, nuint width, nuint featureChannels, MPSImageFeatureChannelFormat featureChannelFormat, IMTLDevice device)
            : this (new MPSImage (device, MPSImageDescriptor.GetImageDescriptor (
                featureChannelFormat, width, height, featureChannels)))
        {
        }

        public MPSImageTensor (int height, int width, int featureChannels = 3, IMTLDevice? device = null)
            : this (new MPSImage (device.Current (), MPSImageDescriptor.GetImageDescriptor (
                MPSImageFeatureChannelFormat.Float32, (nuint)width, (nuint)height, (nuint)featureChannels, 1, MTLTextureUsage.ShaderRead | MTLTextureUsage.ShaderWrite)))
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
            if (texture == null)
                throw new Exception ("Failed to create texture");

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
            if (texture == null)
                throw new Exception ("Failed to create texture");

            image = new MPSImage (texture, (nuint)featureChannels);
            shape = new[] { (int)image.Height, (int)image.Width, (int)image.FeatureChannels };
        }

        //public override Tensor Clone ()
        //{
        //    var device = image.Device;
        //    var newTensor = new MPSImageTensor (image.Height, image.Width, image.FeatureChannels, image.FeatureChannelFormat, device);
        //    var newImage = newTensor.image;
        //    //Console.WriteLine (newImage);
        //    return newTensor;
        //}

        public unsafe override void Copy (Span<float> destination, IMTLDevice device)
        {
            ValidateCopyDestination (destination);
            var dataLayout = MPSDataLayout.HeightPerWidthPerFeatureChannels;
            var dtype = image.PixelFormat;
            switch (dtype) {
                case MTLPixelFormat.R32Float:
                case MTLPixelFormat.RG32Float:
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

        public override MPSImage GetMetalImage (IMTLDevice device)
        {
            if ((image.Device is IMTLDevice d) && device.Handle != d.Handle)
                throw new ArgumentException ($"Cannot get image {Label} for {device.Name} because it was created on {image.Device.Name}");
            if (image.Handle == IntPtr.Zero)
                throw new ObjectDisposedException ($"The metal image for this tensor ({Label}) was disposed after the tensor was created.");
            return image;
        }

        public static (Tensor Left, Tensor Right) CreatePair (NSUrl url, int featureChannels, IMTLDevice? device)
        {
            if (url is null) {
                throw new ArgumentNullException (nameof (url));
            }

            using var pool = new NSAutoreleasePool ();
            var dev = device.Current ();
            using var loader = new MTKTextureLoader (dev);
            using var texture = loader.FromUrl (url, null, out var error);
            error.ValidateNoError ();
            if (texture == null)
                throw new Exception ("Failed to create texture");

            using var image = new MPSImage (texture, (nuint)featureChannels);
            var height = (int)image.Height;
            var width = (int)image.Width / 2;

            using var queue = dev.CreateCommandQueue ();
            if (queue is null)
                throw new Exception ($"Failed to create queue for image pairs");
            var regions = new[] {
                MTLRegion.Create2D (0, 0, width, height),
                MTLRegion.Create2D (width, 0, width, height),
            };
            MPSNNCropAndResizeBilinear lcrop, rcrop;
            unsafe {
                fixed (MTLRegion* regionsP = regions) {
                    lcrop = new MPSNNCropAndResizeBilinear (dev, (nuint)width, (nuint)height, 1, (IntPtr)regionsP);
                    rcrop = new MPSNNCropAndResizeBilinear (dev, (nuint)width, (nuint)height, 1, (IntPtr)(regionsP + 1));
                }
            }
            using var commands = MPSCommandBuffer.Create (queue);
            var halfDesc = MPSImageDescriptor.GetImageDescriptor (image.FeatureChannelFormat, (nuint)width, (nuint)height, image.FeatureChannels);
            var left = new MPSImage (dev, halfDesc);
            var right = new MPSImage (dev, halfDesc);
            lcrop.EncodeToCommandBuffer (commands, image, left);
            rcrop.EncodeToCommandBuffer (commands, image, right);
            commands.Commit ();
            commands.WaitUntilCompleted ();
            commands.Error.ValidateNoError ();

            var leftT = new MPSImageTensor (left);
            var rightT = new MPSImageTensor (right);
            return (leftT, rightT);
        }

        public static (Tensor Left, Tensor Right) CreatePair (string path, int featureChannels, IMTLDevice? device = null)
        {
            return CreatePair (NSUrl.FromFilename (path), featureChannels, device);
        }
    }
}
