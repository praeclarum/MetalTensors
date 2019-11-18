using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using AppKit;
using Foundation;

namespace Tests.Mac
{
    public partial class ViewController : NSViewController
    {
        bool allTestsPassed = false;

        public ViewController (IntPtr handle) : base (handle)
        {
        }

        public override async void ViewDidLoad ()
        {
            base.ViewDidLoad ();

            resultsTextView.Hidden = true;

            await Task.Run (() => {
                try {

                    RunTests ();
                }
                catch (Exception ex) {
                    Console.WriteLine (ex);
                }
            });

            if (allTestsPassed) {
                await Task.Delay (1000);
                NSApplication.SharedApplication.Terminate (this);
            }
        }

        void ShowTestResult (TestResult tr)
        {
            Console.WriteLine (tr);
            if (!tr.Success) {
                var ex = tr.Exception;
                while (ex?.InnerException != null)
                    ex = ex.InnerException;
                Console.WriteLine (ex);

                var elines = (ex.StackTrace ?? "").Split ('\n');
                var e = elines.Length;
                while (e > 0 && SystemLine (elines[e - 1]))
                    e--;
                var s = 0;
                while (s < e && TestLine (elines[s]))
                    s++;
                var glines = elines.Skip (s).Take (e - s).ToList ();
                static bool SystemLine (string line)
                {
                    var t = line.TrimStart ();
                    return t.StartsWith ("at System.") || t.StartsWith ("at (wr");
                }

                static bool TestLine (string line)
                {
                    var t = line.TrimStart ();
                    return t.StartsWith ("at NUnit.");
                }

                glines.Insert (0, ex.Message);
                glines.Insert (0, ex.GetType ().FullName);
                var m = "\n" + string.Join ("\n", glines);
                resultsTextView.Value += m;
                SetOKColor (false);
                resultsTextView.Hidden = false;
            }
        }

        void ShowFinalResult (bool allOK)
        {
            allTestsPassed = allOK;

            var m = $"\n\nALL OK? {allOK}";
            Console.WriteLine (m);
            resultsTextView.Value += m;

            NSWindow? w = this.View.Window;
            if (w != null) {
                w.Title = m.Trim ();
                SetOKColor (allOK);
            }
        }

        void SetOKColor (bool allOK)
        {
            NSWindow? w = this.View.Window;
            w.BackgroundColor = (allOK ? NSColor.Green : NSColor.Red).BlendedColor (0.5f, NSColor.WindowBackground);
        }

        void RunTests ()
        {
            var tests = FindTests ();

            var allOK = true;

            Parallel.ForEach (tests, tf => {
                //Console.WriteLine (System.Threading.Thread.CurrentThread.ManagedThreadId);
                foreach (var t in tf.Tests) {
                    var tr = RunTest (tf, t);
                    allOK = allOK && tr.Success;
                    BeginInvokeOnMainThread (() => ShowTestResult (tr));
                }
            });

            BeginInvokeOnMainThread (() => ShowFinalResult (allOK));
        }

        private TestResult RunTest (TestFixture tf, Test t)
        {
            try {
                var iresult = t.TestMethod.Invoke (tf.TestObject, Array.Empty<object> ());
                return new TestResult (tf, t, null);
            }
            catch (Exception ex) {
                return new TestResult (tf, t, ex);
            }
        }

        class TestResult
        {
            public bool Success => Exception == null;

            public TestFixture Fixture { get; }
            public Test Test { get; }
            public Exception? Exception { get; }

            public TestResult (TestFixture fixture, Test test, Exception? exception)
            {
                Fixture = fixture;
                Test = test;
                Exception = exception;
            }

            public override string ToString ()
            {
                return $"{Fixture}.{Test}() = {Success}";
            }
        }

        class TestFixture
        {
            public object TestObject { get; }
            public Test[] Tests { get; }

            public TestFixture (object testObject, Test[] tests)
            {
                TestObject = testObject;
                Tests = tests;
            }

            public override string ToString () => TestObject.GetType ().FullName;
        }

        class Test
        {
            public MethodInfo TestMethod { get; }

            public Test (MethodInfo method)
            {
                TestMethod = method;
            }

            public override string ToString () => TestMethod.Name;
        }

        private TestFixture[] FindTests ()
        {
            var asmPaths = new[] { Assembly.GetCallingAssembly ().Location };

            var asms = asmPaths.Select (x => Assembly.LoadFile (x));

            var testTypes = asms.SelectMany (FindAsmTests).ToArray ();

            return testTypes.Where (x => x != null).Select (x => x!).ToArray ();

            IEnumerable<TestFixture?> FindAsmTests (Assembly asm)
            {
                return asm.Modules.SelectMany (x => x.GetTypes ()).Select (GetTestType);
            }

            TestFixture? GetTestType (Type type)
            {
                var pmeths = type.GetMethods (BindingFlags.Public | BindingFlags.Instance);

                var tmeths = pmeths.Where (x => HasAttr (x, "TestAttribute")).ToArray ();
                if (tmeths.Length == 0)
                    return null;

                var testo = Activator.CreateInstance (type);
                var tests = tmeths.Select (LoadTest).ToArray ();

                return new TestFixture (testo, tests);

                Test LoadTest (MethodInfo tt)
                {
                    return new Test (tt);
                }
            }

            static bool HasAttr (ICustomAttributeProvider m, string aname)
            {
                return m.GetCustomAttributes (false).Any (x => x.GetType ().Name == aname);
            }
        }

        public override NSObject RepresentedObject {
            get {
                return base.RepresentedObject;
            }
            set {
                base.RepresentedObject = value;
                // Update the view, if already loaded.
            }
        }
    }
}

