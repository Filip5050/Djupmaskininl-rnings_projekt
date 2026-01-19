"""
Main entry point for Credit Card Fraud Detection System
Command-line interface for training, evaluation, and web app
"""
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.train import full_training_pipeline
from src.evaluate import full_evaluation_pipeline
from src.config import Config


def train_command(args):
    """Execute training pipeline"""
    print("\n🚀 Starting fraud detection training...")
    full_training_pipeline()


def evaluate_command(args):
    """Execute evaluation pipeline"""
    print("\n🔍 Evaluating fraud detection model...")
    full_evaluation_pipeline()


def webapp_command(args):
    """Launch Streamlit web application"""
    import subprocess
    print("\n🌐 Launching fraud detection dashboard...")
    subprocess.run(['streamlit', 'run', 'web_app/app.py'])


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='💳 Credit Card Fraud Detection System - Deep Learning Based Fraud Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train

  # Evaluate trained model
  python main.py evaluate

  # Launch web dashboard
  python main.py webapp

For more information, see README.md
        """
    )
    
    subparsers = parser.add_subparsers(title='commands', dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train fraud detection model')
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    eval_parser.set_defaults(func=evaluate_command)
    
    # Web app command
    webapp_parser = subparsers.add_parser('webapp', help='Launch Streamlit fraud detection dashboard')
    webapp_parser.set_defaults(func=webapp_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        print("\n" + "="*70)
        print("💳 QUICK START - FRAUD DETECTION")
        print("="*70)
        print("\n1. Train model:")
        print("   python main.py train")
        print("\n2. Evaluate model:")
        print("   python main.py evaluate")
        print("\n3. Launch dashboard:")
        print("   python main.py webapp")
        print("\n" + "="*70 + "\n")
        sys.exit(0)
    
    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║       💳 CREDIT CARD FRAUD DETECTION SYSTEM 💳           ║
    ║                                                           ║
    ║       Deep Learning Based Fraud Detection                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    main()
