using System;
using System.IO;
using MetalPerformanceShaders;
using MetalTensors;
using MetalTensors.Applications;
using NUnit.Framework;

using static Tests.Imaging;

namespace Tests
{
    public class AutoEncodeTests
    {
        Model MakeEncoder ()
        {
            var input = Tensor.InputImage ("image", 256, 256);
            var encoded =
                input
                .Conv (32, size: 4, stride: 2)
                .LeakyReLU ()
                .Conv (64, size: 4, stride: 2)
                .LeakyReLU ()
                .Conv (128, size: 4, stride: 2)
                .LeakyReLU ()
                .Conv (128, size: 4, stride: 2)
                .LeakyReLU ();
            return encoded.Model (input);
        }

        Model MakeDecoder ()
        {
            var input = Tensor.InputImage ("encoded", 32, 32, 128);
            var decoded =
                input
                .ConvTranspose (128, size: 4, stride: 2)
                .BatchNorm ()
                .ReLU ()
                .ConvTranspose (64, size: 4, stride: 2)
                .BatchNorm ()
                .ReLU ()
                .ConvTranspose (32, size: 4, stride: 2)
                .BatchNorm ()
                .ReLU ()
                .ConvTranspose (32, size: 4, stride: 2)
                .BatchNorm ()
                .ReLU ()
                .Conv (32, size: 4)
                .ReLU ()
                .Conv (3, size: 4)
                .Tanh ();
            return decoded.Model (input);
        }

        Model MakeAutoEncoder ()
        {
            var encoder = MakeEncoder ();
            var decoder = MakeDecoder ();
            var autoEncoder = decoder.Call (encoder);
            return autoEncoder;
        }

        //[Test]
        public void EncoderUntrained ()
        {
            var encoder = MakeEncoder ();
            var output = SaveModelJpeg (encoder, 0.5f, 0.5f);
            Assert.AreEqual (16, output.Shape[0]);
            Assert.AreEqual (16, output.Shape[1]);
            Assert.AreEqual (128, output.Shape[2]);
        }

        [Test]
        public void Train ()
        {
            var autoEncoder = MakeAutoEncoder ();
            autoEncoder.Compile (Loss.MeanAbsoluteError, 1e-4f);
            var uoutput = SaveModelJpeg (autoEncoder, 0.5f, 0.5f, "Untrained");
            var data = GetPix2pixDataSet ();
            var batchSize = 16;
            var batchesPerStep = 50;
            var numSteps = 10;
            var row = 0;
            for (var si = 0; si < numSteps; si++) {
                for (var bi = 0; bi < batchesPerStep; bi++) {
                    var (ins, outs) = data.GetBatch (row, batchSize, autoEncoder.Device.Current ());
                    row = (row + batchSize) % data.Count;
                    var h = autoEncoder.Fit (ins, ins);
                    var aloss = h.AverageLoss;
                    Console.WriteLine ($"AUTOENCODER BATCH {si}/{bi} LOSS {aloss}");
                    h.DisposeSourceImages ();
                }
                var output = SaveModelJpeg (autoEncoder, 0.5f, 0.5f, $"Trained{si}");
                Assert.AreEqual (256, output.Shape[0]);
                Assert.AreEqual (256, output.Shape[1]);
                Assert.AreEqual (3, output.Shape[2]);
            }
        }
    }
}
