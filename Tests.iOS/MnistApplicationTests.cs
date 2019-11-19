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
        public void Train ()
        {
            var app = new MnistApplication ();
            var data = new MnistApplication.MnistDataSet ();

            //app.Train (data);
        }

    }
}
