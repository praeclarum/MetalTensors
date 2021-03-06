﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class PredictTests
    {
        [Test]
        public void AddConstants ()
        {
            var x0 = Tensor.Input ("x0");
            var x1 = Tensor.Constant ("x1", 40);
            var y = x0 + x1;
            var m = y.Model ();

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
            Assert.AreEqual (0, m.Labels.Length);
            Assert.AreEqual (2, m.Sources.Length);
            Assert.AreEqual (1, m.Layers.Length);

            var r = m.Predict (Tensor.Constant (3));

            Assert.AreEqual (43, r[0], 1e-6f);

            //
            // TODO: Graph results should have the right shape
            //
            //Assert.AreEqual (1, r.Shape.Length);
            //Assert.AreEqual (1, r.Shape[0]);
        }
    }
}
