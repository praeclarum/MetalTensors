using Foundation;
using UIKit;
using MonoTouch.NUnit.UI;

namespace Tests.iOS
{
    [Register ("UnitTestAppDelegate")]
    public class UnitTestAppDelegate : UIApplicationDelegate
    {
        UIWindow window;
        TouchRunner runner;

        public override bool FinishedLaunching (UIApplication application, NSDictionary launchOptions)
        {
            window = new UIWindow (UIScreen.MainScreen.Bounds);
            runner = new TouchRunner (window) {
                AutoStart = true
            };

            runner.Add (System.Reflection.Assembly.GetExecutingAssembly ());

            window.RootViewController = new UINavigationController (runner.GetViewController ());

            window.MakeKeyAndVisible ();
            return true;
        }
    }
}
