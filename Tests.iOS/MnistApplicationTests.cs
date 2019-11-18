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
            var app = new MnistApplication ();
            var data = new MnistApplication.DataSet ();
            //app.Train (data);
        }
    }
}
