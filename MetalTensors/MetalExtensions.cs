﻿using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;
using System.Diagnostics;
using System.Threading.Tasks;
using CoreGraphics;

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
            var def = MPSKernel.GetPreferredDevice (MPSDeviceOptions.Default);
            if (def == null || def.Name.Contains ("iOS simulator")) {
                throw new NotSupportedException ("Metal is not supported on this device");
            }
            return def;
        }

        public static void ValidateNoError (this NSError? error)
        {
            if (error != null) {
                throw new NSErrorException (error);
            }
        }

        public static int GetByteSize (this MPSDataType dataType) =>
            dataType switch {
                MPSDataType.Unorm8 => 1,
                MPSDataType.Float32 => 4,
                var x => throw new NotSupportedException ($"Cannot get size of {x}")
            };

        public static void Dispose (this Tensor[][] tensors)
        {
            foreach (var ts in tensors) {
                Dispose (ts);
            }
        }

        private static void Dispose (this Tensor[] tensors)
        {
            foreach (var t in tensors) {
                if (t is Tensors.MPSImageTensor it) {
                    it.MetalImage.Dispose ();
                }
            }
        }

#if __IOS__
        public static void DidModify (this IMTLBuffer buffer, NSRange range)
        {
        }
#endif
    }

    public static class StaticRandom
    {
        private static readonly Random seeder = new Random ();
        [ThreadStatic] private static Random? threadRandom;

        public static int Next ()
        {
            if (threadRandom == null) {
                int seed;
                lock (seeder) {
                    seed = seeder.Next ();
                }
                threadRandom = new Random (seed);
            }
            return threadRandom.Next ();
        }

        public static int Next (int maxValue)
        {
            if (threadRandom == null) {
                int seed;
                lock (seeder) {
                    seed = seeder.Next ();
                }
                threadRandom = new Random (seed);
            }
            return threadRandom.Next (maxValue);
        }

        public static double NextDouble ()
        {
            if (threadRandom == null) {
                int seed;
                lock (seeder) {
                    seed = seeder.Next ();
                }
                threadRandom = new Random (seed);
            }
            return threadRandom.NextDouble ();
        }

        public static double NextNormal ()
        {
            if (threadRandom == null) {
                int seed;
                lock (seeder) {
                    seed = seeder.Next ();
                }
                threadRandom = new Random (seed);
            }
            var u1 = threadRandom.NextDouble ();
            var u2 = threadRandom.NextDouble ();
            return Math.Sqrt (-2.0 * Math.Log (u1)) * Math.Cos (2.0 * Math.PI * u2);
        }
    }
}
