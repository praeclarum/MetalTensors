using System;
using Foundation;
using MetalTensors.Tensors;
using NUnit.Framework;

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
    }
}
