using System;
using MetalTensors.Applications;
using NUnit.Framework;

namespace Tests
{
    public class MnistApplicationTests
    {
        [Test]
        public void LoadData ()
        {
            var data = new MnistApplication.MnistDataSet ();
            Assert.AreEqual (60_000, data.Count);
        }

        [Test]
        public void TrainSmall ()
        {
            var app = new MnistApplication ();

            var batchSize = 5;
            var data = new MnistApplication.MnistDataSet ().Subset (0, 7 * batchSize);

            app.Train (data, batchSize: batchSize, epochs: 3);
        }

    }
}
