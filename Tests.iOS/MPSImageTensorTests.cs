﻿using System;
using Foundation;
using NUnit.Framework;

using MetalTensors;
using MetalTensors.Tensors;

namespace Tests
{
    public class MPSImageTensorTests
    {
        [Test]
        public void FromUrl ()
        {
            var url = NSBundle.MainBundle.GetUrlForResource ("elephant", "jpg");
            var image = new MPSImageTensor (url);
            Assert.AreEqual (3, image.Shape.Length);
            Assert.AreEqual (512, image.Shape[0]);
            Assert.AreEqual (512, image.Shape[1]);
            Assert.AreEqual (3, image.Shape[2]);
        }

        [Test]
        public void CorrectDimensions ()
        {
            var image = Tensor.ImageResource ("rgbywb3x2", "png");
            Assert.AreEqual (2, image.Shape[0]);
            Assert.AreEqual (3, image.Shape[1]);
            Assert.AreEqual (3, image.Shape[2]);
        }

        [Test]
        public void Slice2 ()
        {
            var image = Tensor.ImageResource ("rgbywb3x2", "png");

            AssertColor (  1,   0,   0, image.Slice (0, 0));
            AssertColor (  0,   1,   0, image.Slice (0, 1));
            AssertColor (  0,   0,   1, image.Slice (0, 2));
            AssertColor (  1,   1,   0, image.Slice (1, 0));
            AssertColor (  1,   1,   1, image.Slice (1, 1));
            AssertColor (  0,   0,   0, image.Slice (1, 2));
        }

        void AssertColor (float r, float g, float b, Tensor color)
        {
            Assert.AreEqual (1, color.Shape.Length);
            Assert.AreEqual (3, color.Shape[0]);
            Assert.AreEqual (r, color[0]);
            Assert.AreEqual (g, color[1]);
            Assert.AreEqual (b, color[2]);
        }

        [Test]
        public void Slice3 ()
        {
            var image = Tensor.ImageResource ("rgbywb3x2", "png");

            // Primaries
            Assert.AreEqual (1, image.Slice (0, 0, 0)[0]);
            Assert.AreEqual (1, image.Slice (0, 1, 1)[0]);
            Assert.AreEqual (1, image.Slice (0, 2, 2)[0]);
        }

        [Test]
        public void Slice2Index1 ()
        {
            var image = Tensor.ImageResource ("rgbywb3x2", "png");

            // Yellow
            Assert.AreEqual (1, image.Slice (1, 0)[0]);
            Assert.AreEqual (1, image.Slice (1, 0)[1]);
            Assert.AreEqual (0, image.Slice (1, 0)[2]);
        }

        [Test]
        public void ReadImageResourceIsMPSImage ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            Assert.IsTrue (image is MPSImageTensor);
        }

        [Test]
        public void ReadImageIsMPSImage ()
        {
            var path = NSBundle.MainBundle.PathForResource ("elephant", "jpg");
            var image = Tensor.Image (path);
            Assert.IsTrue (image is MPSImageTensor);
        }
    }
}
