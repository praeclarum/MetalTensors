using System;

namespace MetalTensors.Applications
{
    public class Pix2pixApplication : GanApplication
    {
        // The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1

        const float lambdaL1 = 100.0f;

        public Pix2pixApplication (int height = 256, int width = 256)
            : base (MakeGenerator (height, width),
                    MakeDiscriminator (height, width))
        {
        }

        static Model MakeGenerator (int height, int width)
        {
            var x = Tensor.InputImage ("image", height, width);

            var useDropout = true;
            var instanceNorm = false;
            var ngf = 64;
            var numDowns = 8;

            var unet = UnetSkipConnection (ngf * 8, ngf * 8, inputNC: null, submodule: null, instanceNorm: instanceNorm, innermost: true);
            for (var i = 0; i < numDowns - 5; i++) {
                unet = UnetSkipConnection (ngf * 8, ngf * 8, inputNC: null, submodule: unet, instanceNorm: instanceNorm, useDropout: useDropout);
            }
            unet = UnetSkipConnection (ngf * 4, ngf * 8, inputNC: null, submodule: unet, instanceNorm: instanceNorm);
            unet = UnetSkipConnection (ngf * 2, ngf * 4, inputNC: null, submodule: unet, instanceNorm: instanceNorm);
            unet = UnetSkipConnection (ngf, ngf * 2, inputNC: null, submodule: unet, instanceNorm: instanceNorm);
            unet = UnetSkipConnection (3, ngf, inputNC: null, submodule: unet, outermost: true, instanceNorm: instanceNorm);

            return unet;

            // https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/1c733f5bd7a3aff886403a46baada1db62e2eca8/models/networks.py#L468
            Model UnetSkipConnection (int outerNC, int innerNC, int? inputNC = null, Model? submodule = null, bool outermost = false, bool innermost = false, bool instanceNorm = false, bool useDropout = false)
            {
                var useBias = instanceNorm;
                var inNC = inputNC ?? outerNC;

                var size = submodule != null ? submodule.Output.Shape[0] * 2 : 2;
                var input = Tensor.Input ("image", size, size, inNC);
                var label = "Unet" + size;

                if (outermost) {
                    var downconv = input.Conv (innerNC, size: 4, stride: 2, bias: useBias);
                    var down = downconv;
                    var downsub = submodule != null ? down.Apply (submodule) : down;
                    var up = downsub.ReLU (a: 0).ConvTranspose (outerNC, size: 4, stride: 2, bias: true).Tanh ();
                    return up.Model (label);
                }
                else if (innermost) {
                    var downrelu = input.ReLU (a: 0.2f);
                    var downconv = downrelu.Conv (innerNC, size: 4, stride: 2, bias: useBias);
                    var down = downconv;
                    var up = down.ReLU (a: 0).ConvTranspose (outerNC, size: 4, stride: 2, bias: useBias).BatchNorm ();
                    return input.Concat (up).Model (label);
                }
                else {
                    var downrelu = input.ReLU (a: 0.2f);
                    var downconv = downrelu.Conv (innerNC, size: 4, stride: 2, bias: useBias);
                    var downnorm = downconv.BatchNorm ();
                    var down = downnorm;
                    var downsub = submodule != null ? down.Apply (submodule) : down;
                    var up = downsub.ReLU (a: 0).ConvTranspose (outerNC, size: 4, stride: 2, bias: useBias).BatchNorm ();
                    return input.Concat (up).Model (label);
                }
            }
        }

        static Model MakeDiscriminator (int height, int width)
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
            return input.Conv (channels, size: 4, stride: 2, bias: false).BatchNorm ().ReLU (a: 0.2f);
        }

        static Tensor CD (int channels, Tensor input)
        {
            return input.Conv (channels, size: 4, stride: 2, bias: false).BatchNorm ().Dropout (0.5f).ReLU (a: 0.0f);
        }

        static Tensor CFirst (int channels, Tensor input)
        {
            return input.Conv (channels, size: 4, stride: 2, bias: true).ReLU (a: 0.2f);
        }
    }
}
