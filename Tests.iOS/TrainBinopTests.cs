using System;
using System.Collections.Generic;
using System.Linq;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class TrainBinopTests
    {
        [Test]
        public void Or ()
        {
            TrainBinop ("or", 0.1f, (a, b) => a || b);
        }

        void TrainBinop (string opname, float minLoss, Func<bool, bool, bool> binop)
        {
            var x = Tensor.Input ("x", 2);
            var y = x.Dense (8, biasInit: 0.1f).Tanh ().Dense (1, biasInit: 0.1f).Tanh ();

            var ylabels = Tensor.Labels ("ylabels", 1);

            var loss = y.Loss (ylabels, LossType.MeanSquaredError);

            var rand = new Random ();

            var history = loss.Train (GenTrainingData, learningRate: 0.01f, batchSize: 16, numBatches: 100);

            var batch = history.Batches[^1];
            Assert.AreEqual (1, batch.Loss[0].Shape[0]);

            var belowMinLoss = false;
            for (var bi = 0; bi < history.Batches.Length; bi++) {
                var b = history.Batches[bi];
                var sum = 0.0f;
                var count = 0;
                foreach (var l in b.Loss) {
                    sum += l[0];
                    count++;
                }
                var bl = sum / count;
                //Console.WriteLine ($"BATCH {bi:#,0} LOSS {bl}");
                if (bl < minLoss) {
                    belowMinLoss = true;
                    break;
                }
            }
            Assert.IsTrue (belowMinLoss, "Did not train well");

            IEnumerable<Tensor> GenTrainingData (TensorHandle[] handles)
            {
                var r = new Tensor[handles.Length];

                var x0 = rand.NextDouble ();
                var x1 = rand.NextDouble ();
                var x0b = x0 >= 0.5;
                var x1b = x1 >= 0.5;
                var yb = binop (x0b, x1b);
                var y0 = (yb ? 1.0 : 0.0) + (rand.NextDouble () - 0.5) * 0.001;
                //Console.WriteLine ($"{x0} {opname} {x1} = {y0}");

                for (var i = 0; i < handles.Length; i++) {
                    if (handles[i].Label == "x") {
                        r[i] = Tensor.Array (x0, x1);
                    }
                    else {
                        r[i] = Tensor.Array (y0);
                    }
                }
                return r;
            }
        }
    }
}
