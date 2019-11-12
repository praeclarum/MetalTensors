using System;
using System.Runtime.InteropServices;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public static class MetalExtensions
    {
        static IMTLDevice? currentDevice;

        public static IMTLDevice Current (this IMTLDevice? device)
        {
            if (device != null)
                return device;
            if (currentDevice != null)
                return currentDevice;
            var def = Default (null);
            var old = Interlocked.CompareExchange (ref currentDevice, def, null);
            if (old == null) {
                Console.WriteLine ("DEVICE = " + def.Name);
            }
            return currentDevice;
        }

        static IMTLDevice Default (this IMTLDevice? device)
        {
            if (device != null)
                return device;
            var def = MTLDevice.SystemDefault;
            if (def == null || def.Name.Contains ("iOS simulator")) {
                throw new NotSupportedException ("Metal is not supported on this device");
            }
            return def;
        }

        public static void ValidateNoError (this NSError? error)
        {
            if (error != null) {
                throw new Exception (error.ToString ());
            }
        }

        public static int GetShapeLength (this int[] shape)
        {
            var r = 1;
            for (var i = 0; i < shape.Length; i++) {
                r *= shape[i];
            }
            return r;
        }

        public static unsafe void Fill (this MPSImage image, float constant)
        {
            var dtype = image.PixelFormat;
            var dataLayout = MPSDataLayout.HeightPerWidthPerFeatureChannels;

            switch (dtype) {
                case MTLPixelFormat.R32Float: {
                        var len = (int)(image.Height * image.Width * image.FeatureChannels);
                        Span<float> dataSpan = len < 1024 ?
                            stackalloc float[len] :
                            new float[len];
                        for (var i = 0; i < len; i++) {
                            dataSpan[i] = constant;
                        }
                        fixed (float* dataPtr = dataSpan) {
                            image.WriteBytes ((IntPtr)dataPtr, dataLayout, 0);
                        }
                    }
                    break;
                default:
                    throw new NotSupportedException ($"Cannot fill images with pixel format {dtype}");
            }
        }

        public static MPSVectorDescriptor VectorDescriptor (int length, MPSDataType dataType = MPSDataType.Float32) =>
            MPSVectorDescriptor.Create ((nuint)length, dataType);

        public static MPSVector Vector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
        {
            if (descriptor.Length <= 0)
                throw new ArgumentOutOfRangeException (nameof (descriptor), "Vector lengths must be > 0");

            var v = new MPSVector (device, descriptor);
            if (v.Data == null)
                throw new Exception ($"Failed to create vector with length {descriptor.Length}");
            var vectorByteSize = GetByteSize (descriptor);
            if (vectorByteSize > 0) {
                unsafe {
                    float biasInit = initialValue;
                    var biasInitPtr = (IntPtr)(float*)&biasInit;
                    memset_pattern4 (v.Data.Contents, biasInitPtr, vectorByteSize);
                }
            }
            return v;
        }
        [System.Runtime.InteropServices.DllImport (@"__Internal", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
        static extern void memset_pattern4 (IntPtr b, IntPtr pattern4, nint len);

        public static float[] ToArray (this MPSVector vector)
        {
            var ar = new float[vector.Length];
            Marshal.Copy (vector.Data.Contents, ar, 0, ar.Length);
            return ar;
        }

        public static bool IsValid (this MPSVector vector)
        {
            var ar = vector.ToArray ();
            for (var i = 0; i < ar.Length; i++) {
                var v = ar[i];
                if (float.IsNaN (v))
                    return false;
                if (float.IsInfinity (v))
                    return false;
                if (float.IsNegativeInfinity (v))
                    return false;
            }
            return true;
        }

        public static int GetByteSize (this MPSVectorDescriptor descriptor) =>
            (int)descriptor.Length * GetByteSize (descriptor.DataType);

        public static int GetByteSize (this MPSDataType dataType) =>
            dataType switch
            {
                MPSDataType.Unorm8 => 1,
                MPSDataType.Float32 => 4,
                var x => throw new NotSupportedException ($"Cannot get size of {x}")
            };
#if __IOS__
        public static void DidModify (this IMTLBuffer buffer, NSRange range)
        {
        }
#endif
    }
}
