using System;
using MetalTensors.Applications;
using NUnit.Framework;

namespace Tests
{
    public class Pix2pixApplicationTests
    {
        [Test]
        public void DefaultShapes ()
        {
            var pix2pix = new Pix2pixApplication ();

            Assert.AreEqual (256, pix2pix.Generator.Input.Shape[0]);
            Assert.AreEqual (256, pix2pix.Generator.Input.Shape[1]);
            Assert.AreEqual (3, pix2pix.Generator.Input.Shape[2]);

            //Assert.AreEqual (1, pix2pix.Discriminator.Output.Shape[0]);
            //Assert.AreEqual (1, pix2pix.Discriminator.Output.Shape[1]);
            //Assert.AreEqual (1, pix2pix.Discriminator.Output.Shape[2]);

            Assert.NotNull (pix2pix.Gan);

            //Assert.AreEqual (1, pix2pix.Gan.Output.Shape[0]);
            //Assert.AreEqual (1, pix2pix.Gan.Output.Shape[1]);
            //Assert.AreEqual (1, pix2pix.Gan.Output.Shape[2]);
        }
    }
}
