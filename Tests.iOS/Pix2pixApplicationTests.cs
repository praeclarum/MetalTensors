using System;
using System.IO;
using MetalTensors;
using MetalTensors.Applications;
using NUnit.Framework;

using static Tests.Imaging;

namespace Tests
{
    public class Pix2pixApplicationTests
    {
        [Test]
        public void ASimpleGenerator ()
        {
            var input = Tensor.InputImage ("image", 256, 256);
            var gen = input.Conv (3, size: 4).Tanh();
            var output = SaveModelJpeg (input, gen, 0.5f, 0.5f);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);
        }

        [Test]
        public void ADownsamplingGenerator ()
        {
            var input = Tensor.InputImage ("image", 256, 256);
            var gen = input.Conv (32, size: 4, stride: 2).ReLU().ConvTranspose(3, size: 4, stride:2).Tanh();
            var output = SaveModelJpeg (input, gen, 0.5f, 0.5f);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);
        }

        [Test]
        public void ADownsamplingGeneratorWithSkip ()
        {
            var input = Tensor.InputImage ("image", 256, 256);
            var gen = input
                .Conv (32, size: 4, stride: 2)
                .ReLU()
                .ConvTranspose(32, size: 4, stride:2)
                .BatchNorm ()
                .LeakyReLU ()
                .Concat (input)
                .Conv (3, size:4)
                .Tanh();
            var output = SaveModelJpeg (input, gen, 0.5f, 0.5f);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);
        }

        [Test]
        public void ADownsamplingGeneratorModel ()
        {
            var input = Tensor.InputImage ("image", 256, 256);

            var minput = Tensor.InputImage ("image", 256, 256, 32);
            var moutput =
                minput
                .Conv (32, size: 4, stride: 2)
                .ReLU ()
                .ConvTranspose (32, size: 4, stride: 2);
            var innerModel = moutput.Model (minput, "Inner Model");

            var gen =
                innerModel.Call(input.Conv (32, size: 4, stride: 1))
                .Conv (3, size:4)
                .Tanh();
            var output = SaveModelJpeg (input, gen, 0.5f, 0.5f);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);
        }

        [Test]
        public void ADownsamplingNestedModelGenerator ()
        {
            var input = Tensor.InputImage ("image", 256, 256);

            Model MakeInnerModel()
            {
                var minput = Tensor.InputImage ("image", 128, 128, 32);
                var moutput =
                    minput
                    .Conv (32, size: 4, stride: 2)
                    .LeakyReLU ()
                    .ConvTranspose (32, size: 4, stride: 2);
                return moutput.Model (minput, $"Inner Model");
            }

            var innerModel = MakeInnerModel ();

            Model MakeOuterModel ()
            {
                var minput = Tensor.InputImage ("image", 256, 256, 32);
                var moutput =
                    innerModel
                    .Call(minput.Conv (32, size: 4, stride: 2).LeakyReLU ())
                    .ConvTranspose (32, size: 4, stride: 2);
                return moutput.Model (minput, $"Outer Model");
            }

            var outerModel = MakeOuterModel ();

            var gen =
                outerModel
                .Call(input.Conv (32, size: 4, stride: 1).ReLU())
                .Conv (3, size:4)
                .Tanh();
            var output = SaveModelJpeg (input, gen, 0.5f, 0.5f);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);
        }

        [Test]
        public void DefaultShapes ()
        {
            var pix2pix = new Pix2pixApplication ();

            Assert.AreEqual (256, pix2pix.Generator.Input.Shape[0]);
            Assert.AreEqual (256, pix2pix.Generator.Input.Shape[1]);
            Assert.AreEqual (3, pix2pix.Generator.Input.Shape[2]);

            Assert.AreEqual (3, pix2pix.Discriminator.Output.Shape.Length);
            Assert.AreEqual (1, pix2pix.Discriminator.Output.Shape[^1]);

            //Assert.NotNull (pix2pix.Gan);
            //Assert.AreEqual (3, pix2pix.Gan.Output.Shape.Length);
            //Assert.AreEqual (1, pix2pix.Gan.Output.Shape[^1]);
        }

        [Test]
        public void DataSetHasImages ()
        {
            var data = GetPix2pixDataSet ();
            var image = data.GetPairedRow (0);
            image.SaveImage (JpegUrl ());
        }

        [Test]
        public void DataSetHasLeftAndRight ()
        {
            var data = GetPix2pixDataSet ();
            var (inputs, outputs) = data.GetRow (0, MetalExtensions.Current(null));
            inputs[0].SaveImage (JpegUrl (name: "Left"), 0.5f, 0.5f);
            outputs[0].SaveImage (JpegUrl (name: "Right"), 0.5f, 0.5f);
        }

        [Test]
        public void GeneratorOutputsImages ()
        {
            var pix2pix = new Pix2pixApplication ();

            var data = GetPix2pixDataSet ();
            var (inputs, outputs) = data.GetRow (0, pix2pix.Device);
            var output = pix2pix.Generator.Predict (inputs[0], pix2pix.Device);
            output.SaveImage (JpegUrl (), 0.5f, 0.5f);

            Assert.AreEqual (pix2pix.Generator.Output.Shape[0], output.Shape[0]);
            Assert.AreEqual (pix2pix.Generator.Output.Shape[1], output.Shape[1]);
            Assert.AreEqual (pix2pix.Generator.Output.Shape[2], output.Shape[2]);
        }

        [Test]
        public void DiscriminatorOutputsPatches ()
        {
            var pix2pix = new Pix2pixApplication ();

            var data = GetPix2pixDataSet ();
            var (inputs, outputs) = data.GetRow (0, pix2pix.Device);

            var output = pix2pix.Discriminator.Predict (outputs[0], pix2pix.Device);

            output.SaveImage (JpegUrl ());

            Assert.AreEqual (pix2pix.Discriminator.Output.Shape[0], output.Shape[0]);
            Assert.AreEqual (pix2pix.Discriminator.Output.Shape[1], output.Shape[1]);
            Assert.AreEqual (pix2pix.Discriminator.Output.Shape[2], output.Shape[2]);
        }

        [Test]
        public void Train ()
        {
            var pix2pix = new Pix2pixApplication ();

            var data = GetPix2pixDataSet (b2a: true);

            SampleModel ("Train0");
            var lastP = 0.0;
            var (imageCount, trainTime, dataTime) = pix2pix.Train (data, batchSize: 16, epochs: 100.1f, progress: p => {
                //Console.WriteLine ($"Pix2pix {Math.Round (p * 100, 2)}%");
                if ((p - lastP) >= 0.02) {
                    lastP = p;
                    //SampleModel ($"Train{(int)(p * 100)}");
                }
            });

            var trainImagesPerSecond = imageCount / (trainTime.TotalSeconds);
            var dataImagesPerSecond = imageCount / (dataTime.TotalSeconds);
            var totalImagesPerSecond = imageCount / (trainTime.TotalSeconds + dataTime.TotalSeconds);

            Console.WriteLine ($"{imageCount} images in {trainTime + dataTime}");
            Console.WriteLine ($"{trainImagesPerSecond} TrainImages/sec");
            Console.WriteLine ($"{dataImagesPerSecond} DataImages/sec");
            Console.WriteLine ($"{totalImagesPerSecond} Images/sec");
            Console.WriteLine ($"{TimeSpan.FromSeconds (data.Count / totalImagesPerSecond)}/epoch");

            SampleModel ("Trained");

            void SampleModel (string postfix)
            {
                for (var row = 0; row < 1; row++) {
                    var (inputs, outputs) = data.GetRow (row, pix2pix.Device);
                    var goutput = pix2pix.Generator.Predict (inputs[0], pix2pix.Device);
                    goutput.SaveImage (JpegUrl (name: $"Gen{row}{postfix}"), 0.5f, 0.5f);
                    Assert.AreEqual (pix2pix.Generator.Output.Shape[0], goutput.Shape[0]);
                    Assert.AreEqual (pix2pix.Generator.Output.Shape[1], goutput.Shape[1]);
                    Assert.AreEqual (pix2pix.Generator.Output.Shape[2], goutput.Shape[2]);
                    var droutput = pix2pix.Discriminator.Predict (outputs[0], pix2pix.Device);
                    //Console.WriteLine ($"REAL DISCR {droutput.Format ()}");
                    droutput.SaveImage (PngUrl (name: $"DiscrReal{row}{postfix}"));
                    Assert.AreEqual (pix2pix.Discriminator.Output.Shape[0], droutput.Shape[0]);
                    Assert.AreEqual (pix2pix.Discriminator.Output.Shape[1], droutput.Shape[1]);
                    Assert.AreEqual (pix2pix.Discriminator.Output.Shape[2], droutput.Shape[2]);
                    var dfoutput = pix2pix.Discriminator.Predict (goutput, pix2pix.Device);
                    dfoutput.SaveImage (PngUrl (name: $"DiscrFake{row}{postfix}"));
                }
            }
        }
    }
}
