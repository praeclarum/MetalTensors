using System;

namespace MetalTensors.Applications
{
    public class Pix2pixApplication : GanApplication
    {
        public Pix2pixApplication (int height = 256, int width = 256)
            : base (MakeGenerator (height, width),
                    MakeDiscriminator (height, width))
        {
        }

        private static Model MakeGenerator (int height, int width)
        {
            var x = Tensor.InputImage ("image", height, width);

            var e0 = CFirst (64, x);
            var e1 = C (128, e0);
            var e2 = C (256, e1);
            var e3 = C (512, e2);
            var e4 = C (512, e3);
            var e5 = C (512, e4);
            var e6 = C (512, e5);
            var e7 = C (512, e6); // 1x1x512

            var d0 = D (512, e7);

            return d0.Model ();

            Tensor D (int c, Tensor e)
            {
                return CD (c, e);
            }
        }

        private static Model MakeDiscriminator (int height, int width)
        {
            var x = Tensor.InputImage ("image", height, width);

            x = CFirst (64, x);
            x = C (128, x);
            x = C (256, x);
            x = C (512, x);

            // Need to average to get 1x1x1

            return x.Model ();
        }

        static Tensor C (int channels, Tensor input)
        {
            return input.Conv (channels, size: 4, stride: 2).BatchNorm ().ReLU (a: 0.2f);
        }

        static Tensor CD (int channels, Tensor input)
        {
            return input.Conv (channels, size: 4, stride: 2).BatchNorm ().Dropout (0.5f).ReLU (a: 0.0f);
        }

        static Tensor CFirst (int channels, Tensor input)
        {
            return input.Conv (channels, size: 4, stride: 2).ReLU (a: 0.2f);
        }
    }
}
