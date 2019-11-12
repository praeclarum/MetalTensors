using System;
using Foundation;
using NUnit.Framework;

using MetalTensors;
using MetalTensors.Tensors;

namespace Tests.iOS
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
        public void ReadImageIsMPSImage ()
        {
            var path = NSBundle.MainBundle.PathForResource ("elephant", "jpg");
            var image = Tensor.ReadImage (path);
            Assert.IsInstanceOfType (typeof (MPSImageTensor), image);
        }
    }
}
