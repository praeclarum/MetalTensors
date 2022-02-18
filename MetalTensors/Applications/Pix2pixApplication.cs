using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Metal;
using MetalTensors.Tensors;

namespace MetalTensors.Applications
{
    public class Pix2pixApplication : Application
    {
        // The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1

        const float lambdaL1 = 100.0f;

        const float gLearningRate = 0.0002f;
        const float dLearningRate = 0.0004f;

        readonly IMTLDevice device;

        public IMTLDevice Device => device;
        public Model Generator { get; }
        public Model Discriminator { get; }
        public Model Gan { get; }

        bool compiled = false;

        public Pix2pixApplication (int height = 256, int width = 256, IMTLDevice? device = null)
        {
            this.device = device.Current ();
            Generator = CreateGenerator ();
            Discriminator = CreateDiscriminator (height, width);
            Gan = Discriminator.Call (Generator);
        }

        void CompileIfNeeded ()
        {
            if (compiled)
                return;
            compiled = true;
            Discriminator.Compile (Loss.SumSigmoidCrossEntropy, new AdamOptimizer (learningRate: dLearningRate));
            Discriminator.IsTrainable = false;
            Gan.Compile (Loss.SumSigmoidCrossEntropy, new AdamOptimizer (learningRate: gLearningRate));
        }

        static Model CreateGenerator ()
        {
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

                var size = submodule != null ? submodule.Outputs[0].Shape[0] * 2 : 2;
                var input = Tensor.Input ("image", size, size, inNC);
                var name = "Unet" + size;

                if (outermost) {
                    var downconv = input.Conv (innerNC, size: 4, stride: 2, bias: useBias);
                    var down = downconv;
                    var downsub = submodule != null ? down.Apply (submodule) : down;
                    var up = downsub.ReLU ().ConvTranspose (outerNC, size: 4, stride: 2, bias: true).Tanh ();
                    //var up = downsub.ReLU ().Upsample ().Conv (outerNC, size: 4, stride: 1, bias: true).Tanh ();
                    return up.Model (input, "Generator");
                }
                else if (innermost) {
                    var downrelu = input.LeakyReLU (a: 0.2f);
                    var downconv = downrelu.Conv (innerNC, size: 4, stride: 2, bias: useBias);
                    var down = downconv;
                    var up = down.ReLU ().ConvTranspose (outerNC, size: 4, stride: 2, bias: useBias).BatchNorm ();
                    //var up = down.ReLU ().Upsample ().Conv (outerNC, size: 4, stride: 1, bias: useBias).BatchNorm ();
                    return input.Concat (up).Model (input, name);
                }
                else {
                    var downrelu = input.LeakyReLU (a: 0.2f);
                    var downconv = downrelu.Conv (innerNC, size: 4, stride: 2, bias: useBias);
                    var downnorm = downconv.BatchNorm ();
                    var down = downnorm;
                    var downsub = submodule != null ? down.Apply (submodule) : down;
                    var up = downsub.ReLU ().ConvTranspose (outerNC, size: 4, stride: 2, bias: useBias).BatchNorm ();
                    //var up = downsub.ReLU ().Upsample ().Conv (outerNC, size: 4, stride: 1, bias: useBias).BatchNorm ();
                    var updrop = useDropout ? up.Dropout (0.5f) : up;
                    return input.Concat (updrop).Model (input, name);
                }
            }
        }

        static Model CreateDiscriminator (int height, int width)
        {
            // https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/1c733f5bd7a3aff886403a46baada1db62e2eca8/models/networks.py#L538

            var image = Tensor.InputImage ("image", height, width);

            var instanceNorm = false;
            var useBias = instanceNorm;
            var nlayers = 3;
            var kw = 4;
            var ndf = 64;

            var disc =
                image
                .Conv (ndf, size: kw, stride: 2)
                .LeakyReLU (a: 0.2f);

            int nf_mult;
            for (var n = 1; n < nlayers; n++) {
                nf_mult = Math.Min (1 << n, 8);
                disc =
                    disc
                    .Conv (ndf * nf_mult, size: kw, stride: 2, bias: useBias)
                    .BatchNorm ()
                    .LeakyReLU (a: 0.2f);
            }

            nf_mult = Math.Min (1 << nlayers, 8);
            disc =
                disc
                .Conv (ndf * nf_mult, size: kw, stride: 1, bias: useBias)
                .BatchNorm ()
                .LeakyReLU (a: 0.2f);

            disc = disc.Conv (1, size: kw, stride: 1, bias: true);

            disc = disc.Sigmoid ();

            return disc.Model (image, "Discriminator");
        }

        public (int TrainedImages, TimeSpan TrainingTime, TimeSpan DataSetTime) Train (Pix2pixDataSet dataSet, int batchSize = 1, float epochs = 200, Action<double>? progress = null)
        {
            CompileIfNeeded ();

            var trainSW = new Stopwatch ();
            var dataSW = new Stopwatch ();

            var trainImageCount = dataSet.Count;

            var numBatchesPerEpoch = trainImageCount / batchSize;
            var numBatchesToTrain = (int)(epochs * numBatchesPerEpoch) + 1;
            var numImagesToTrain = numBatchesToTrain * batchSize;

            var ones = Tensor.Ones (Discriminator.Output.Shape);
            var zeros = Tensor.Zeros (Discriminator.Output.Shape);
            var zerosBatch = new Tensor[batchSize][];
            var zerosAndOnesBatch = new Tensor[batchSize * 2][];
            for (var i = 0; i < batchSize; i++) {
                zerosBatch[i] = new[] { zeros };
                zerosAndOnesBatch[i] = zerosBatch[i];
            }
            for (var i = 0; i < batchSize; i++) {
                zerosAndOnesBatch[batchSize + i] = new[] { ones };
            }

            var numTrainedImages = 0;

            for (var batch = 0; batch < numBatchesToTrain; batch++) {
                dataSW.Start ();
                var (segments, reals) = dataSet.GetBatch (batch*batchSize, batchSize, device);
                dataSW.Stop ();
                trainSW.Start ();
                var fakes = Generator.Predict (segments);
                var realsAndFakes = reals.Concat(fakes).ToArray ();
                var dh = Discriminator.Fit (realsAndFakes, zerosAndOnesBatch);
                Console.WriteLine ($"PIX2PIX B{batch+1}/{numBatchesToTrain} DLOSS {dh.AverageLoss}");
                Gan.Fit (segments, zerosBatch);
                trainSW.Stop ();
                numTrainedImages += segments.Length;
                progress?.Invoke ((double)numTrainedImages / (double)numImagesToTrain);
            }

            return (numTrainedImages, trainSW.Elapsed, dataSW.Elapsed);
        }

        public class Pix2pixDataSet : DataSet
        {
            private readonly string[] filePaths;
            private readonly bool b2a;

            public override int Count => filePaths.Length;

            public Pix2pixDataSet (string[] filePaths, bool b2a)
            {
                this.filePaths = filePaths;
                this.b2a = b2a;
            }

            public Tensor GetPairedRow (int index)
            {
                return Tensor.Image (filePaths[index]);
            }

            public override (Tensor[] Inputs, Tensor[] Outputs) GetRow (int index, IMTLDevice device)
            {
                var path = filePaths[index];
                var (left, right) = Tensor.ImagePair (path, channelScale: 2.0f, channelOffset: -1.0f, device: device);
                if (b2a)
                    return (new[] { right }, new[] { left });
                return (new[] { left }, new[] { right });
                //return (new[] { Tensor.Zeros (256, 256, 3) }, new[] { Tensor.Ones (256, 256, 3) });
            }

            public static Pix2pixDataSet LoadDirectory (string path, bool b2a = false)
            {
                var files = Directory.GetFiles (path, "*.jpg").OrderBy(x => x).ToArray ();
                return new Pix2pixDataSet (files, b2a);
            }
        }
    }
}
