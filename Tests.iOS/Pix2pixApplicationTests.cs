using System;
using System.IO;
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

            Assert.AreEqual (1, pix2pix.Discriminator.Output.Shape.Length);
            Assert.AreEqual (1, pix2pix.Discriminator.Output.Shape[0]);

            Assert.NotNull (pix2pix.Gan);

            Assert.AreEqual (1, pix2pix.Gan.Output.Shape.Length);
            Assert.AreEqual (1, pix2pix.Gan.Output.Shape[0]);

            Assert.AreEqual (1, pix2pix.Gan.Output.Shape.Length);
            Assert.AreEqual (1, pix2pix.Gan.Output.Shape[0]);
        }


        [Test]
        public void Train ()
        {
            var pix2pix = new Pix2pixApplication ();

            var userDir = Environment.GetFolderPath (Environment.SpecialFolder.MyDocuments);
            var dataDir = Path.Combine (userDir, "Data", "datasets", "facades");
            var trainDataDir = Path.Combine (dataDir, "train");

            var data = Pix2pixApplication.Pix2pixDataSet.LoadDirectory (trainDataDir);

            pix2pix.Train (data, epochs: 2);
        }
    }
}
