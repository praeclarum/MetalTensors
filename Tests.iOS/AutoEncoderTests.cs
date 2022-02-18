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
                .Conv (256, size: 4, stride: 2)
                .LeakyReLU ();
            return encoded.Model (input);
        }

        Model MakeDecoder ()
        {
            var input = Tensor.InputImage ("encoded", 16, 16, 256);
            var decoded =
                input
                //.ConvTranspose (128, size: 4, stride: 2)
                .Conv (128, size: 4)
                .ReLU ()
                .Upsample ().Conv (128, size: 4)
                //.BatchNorm ()
                .ReLU ()
                //.ConvTranspose (64, size: 4, stride: 2)
                .Upsample ().Conv (64, size: 4)
                //.BatchNorm ()
                .ReLU ()
                //.ConvTranspose (32, size: 4, stride: 2)
                .Upsample ().Conv (32, size: 4)
                //.BatchNorm ()
                .ReLU ()
                //.ConvTranspose (32, size: 4, stride: 2)
                .Upsample ().Conv (32, size: 4)
                //.BatchNorm ()
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
            autoEncoder.Compile (Loss.MeanAbsoluteError, 1e-3f);
            SaveModelJpeg (autoEncoder, 0.5f, 0.5f, "Untrained");
            var data = GetIdentityDataSet (b2a: false);
            var batchSize = 16;
            var batchesPerEpoch = data.Count / batchSize;
            var numEpochs = 2;
            var row = 0;
            for (var si = 0; si < numEpochs; si++) {
                for (var bi = 0; bi < batchesPerEpoch; bi++) {
                    var (ins, outs) = data.GetBatch (row, batchSize, autoEncoder.Device.Current ());
                    row = (row + batchSize) % data.Count;
                    var h = autoEncoder.Fit (ins, outs);
                    var aloss = h.AverageLoss;
                    Console.WriteLine ($"AUTOENCODER BATCH E{si + 1} B{bi + 1}/{batchesPerEpoch} LOSS {aloss}");
                    h.DisposeSourceImages ();
                    ins.Dispose ();
                    outs.Dispose ();
                }
                //var h = autoEncoder.Fit (data, batchSize: batchSize, epochs: 1);
                //Console.WriteLine ($"AUTOENCODER E{si + 1} LOSS {h.Batches[^1].AverageLoss}");
                var output = SaveModelJpeg (autoEncoder, 0.5f, 0.5f, $"Trained{si+1}");
                Assert.AreEqual (256, output.Shape[0]);
                Assert.AreEqual (256, output.Shape[1]);
                Assert.AreEqual (3, output.Shape[2]);
                GC.Collect ();
                GC.WaitForPendingFinalizers ();
            }
        }
    }
}
