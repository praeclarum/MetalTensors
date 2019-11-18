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
            var data = new MnistApplication.DataSet ();
            Assert.AreEqual (60_000, data.Length);
        }

        [Test]
        public void Train ()
        {
            var app = new MnistApplication ();
            var data = new MnistApplication.DataSet ();
            //app.Train (data);
        }

    }
}
