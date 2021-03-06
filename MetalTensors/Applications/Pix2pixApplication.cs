﻿using System;
using System.Collections.Generic;
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

        public Model Generator { get; }
        public Model Discriminator { get; }
        public Model Gan { get; }

        public Pix2pixApplication (int height = 256, int width = 256)
        {
            var generator = CreateGenerator (height, width);
            var genOut = generator.Outputs[0];
            Generator = generator;

            var discriminator = CreateDiscriminator (height, width);
            var discOut = discriminator.Outputs[0];
            var discLabels = Tensor.Labels ("discLabels", discOut.Shape);
            var discLoss = discOut.Loss (discLabels, LossType.SigmoidCrossEntropy, ReductionType.Mean);
            Discriminator = discLoss.Model (discriminator.Label);

            var gan = discriminator.Lock ().Apply (generator);
            var ganOut = gan.Outputs[0];
            var genLabels = Tensor.Labels ("genLabels", genOut.Shape);
            var ganLossD = ganOut.Loss (discLabels, LossType.SigmoidCrossEntropy, ReductionType.Mean);
            //var ganLossL1 = genOut.Loss (genLabels, LossType.MeanAbsoluteError, ReductionType.Sum);
            var ganLoss = ganLossD;// + lambdaL1 * ganLossL1;
            Gan = ganLoss.Model (gan.Label);
        }

        static Model CreateGenerator (int height, int width)
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
                    var updrop = useDropout ? up.Dropout (0.5f) : up;
                    return input.Concat (updrop).Model (label);
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

            var disc = image.Conv (ndf, size: kw, stride: 2).ReLU (a: 0.2f);

            int nf_mult;
            for (var n = 1; n < nlayers; n++) {
                nf_mult = Math.Min (1 << n, 8);
                disc = disc.Conv (ndf * nf_mult, size: kw, stride: 2, bias: useBias).BatchNorm ().ReLU (a: 0.2f);
            }

            nf_mult = Math.Min (1 << nlayers, 8);
            disc = disc.Conv (ndf * nf_mult, size: kw, stride: 1, bias: useBias).BatchNorm ().ReLU (a: 0.2f);

            disc = disc.Conv (1, size: kw, stride: 1, bias: true);

            return disc.Model ();
        }

        public void Train (Pix2pixDataSet dataSet, int batchSize = 1, int epochs = 200, IMTLDevice? device = null)
        {
            var trainImageCount = dataSet.Count;

            var numBatchesPerEpoch = trainImageCount / batchSize;

            for (var epoch = 0; epoch < epochs; epoch++) {
                for (var batch = 0; batch < numBatchesPerEpoch; batch++) {
                    //var discHistoryFake = Discriminator.Train (dataSet.LoadData, 0.0002f, batchSize: batchSize, numBatches: numBatchesPerEpoch, device);
                    var index = batch * batchSize;
                    var subdata = dataSet.Subset (index, batchSize);
                    //var discHistoryReal = Discriminator.Train (subdata, 0.0002f, batchSize: batchSize, epochs: 1, device: device);
                    var ganHistory = Gan.Train (subdata, 0.0002f, batchSize: batchSize, epochs: 1, device: device);
                }
            }
        }

        public class Pix2pixDataSet : DataSet
        {
            static readonly string[] cols = { "image", "discLabels" };
            private readonly string[] filePaths;

            public override int Count => filePaths.Length;
            public override string[] Columns => cols;

            public Pix2pixDataSet (string[] filePaths)
            {
                this.filePaths = filePaths;
            }

            public override Tensor[] GetRow (int index)
            {
                return new[] { Tensor.Zeros (), Tensor.Ones () };
            }

            public static Pix2pixDataSet LoadDirectory (string path)
            {
                var files = Directory.GetFiles (path, "*.jpg").OrderBy(x => x).ToArray ();
                return new Pix2pixDataSet (files);
            }
        }
    }
}
