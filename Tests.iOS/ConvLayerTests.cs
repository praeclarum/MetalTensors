using System;
using MetalTensors;
using NUnit.Framework;

using static Tests.Imaging;

namespace Tests
{
    public class ConvLayerTests
    {
        [Test]
        public void Defaults ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            var conv = image.Conv (32, 3);

            Assert.AreEqual (3, conv.Shape.Length);
            Assert.AreEqual (512, conv.Shape[0]);
            Assert.AreEqual (512, conv.Shape[1]);
            Assert.AreEqual (32, conv.Shape[2]);

            Assert.IsTrue (conv[0,0,0] > -10.0f);
        }

        [Test]
        public void ResBlock ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            var conv1 = image.Conv (32, 1);
            var conv2 = conv1.Conv (32, 3);
            var output = conv1 + conv2;

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (512, output.Shape[0]);
            Assert.AreEqual (512, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);

            Assert.IsTrue (output[0, 0, 0] > -10.0f);
        }

        [Test]
        public void ConvStride1 ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.Conv (32, 3, stride: 1).Add(0.5f);
            var output = SaveModelJpeg (image, conv);
            Assert.AreEqual (512, output.Shape[0]);
            Assert.AreEqual (512, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvStride2 ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.Conv (32, 3, stride: 2);
            var output = SaveModelJpeg (image, conv.Add (0.5f));
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvStride2LeakyReLU ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.Conv (32, 3, stride: 2).LeakyReLU (a: 0.2f);
            var output = SaveModelJpeg (image, conv.Add (0.5f));
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvStride2LeakyReLUConvStride2 ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.Conv (32, 3, stride: 2).LeakyReLU (a: 0.2f).Conv (32, 3, stride: 2);
            var output = SaveModelJpeg (image, conv.Add (0.5f));
            Assert.AreEqual (128, output.Shape[0]);
            Assert.AreEqual (128, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvStride2LeakyReLUConvStride2BatchNorm ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.Conv (32, 3, stride: 2).LeakyReLU (a: 0.2f).Conv (32, 3, stride: 2).BatchNorm ();
            var output = SaveModelJpeg (image, conv.Add (0.5f));
            Assert.AreEqual (128, output.Shape[0]);
            Assert.AreEqual (128, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvStride2LeakyReLUConvStride2BatchNormLeakyReLU ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.Conv (32, 3, stride: 2).LeakyReLU (a: 0.2f).Conv (32, 3, stride: 2).BatchNorm ().LeakyReLU (a: 0.2f);
            var output = SaveModelJpeg (image, conv.Add (0.5f));
            Assert.AreEqual (128, output.Shape[0]);
            Assert.AreEqual (128, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvTransposeStride1 ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.ConvTranspose (32, 3, stride: 1).Add (0.5f);
            var output = SaveModelJpeg (image, conv);
            Assert.AreEqual (512, output.Shape[0]);
            Assert.AreEqual (512, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }

        [Test]
        public void ConvTransposeStride2 ()
        {
            var image = Tensor.InputImage ("image", 512, 512, 3);
            var conv = image.ConvTranspose (32, 3, stride: 2).Add (0.5f);
            var output = SaveModelJpeg (image, conv);
            Assert.AreEqual (1024, output.Shape[0]);
            Assert.AreEqual (1024, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);
        }
    }
}
