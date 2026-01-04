from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Run stroke detection fusion pipeline (developer only)'

    def handle(self, *args, **options):
        self.stdout.write('Starting fusion.main()...')
        try:
            from stroke_detection_app import fusion
            fusion.main()
            self.stdout.write(self.style.SUCCESS('fusion.main() finished.'))
        except Exception as e:
            self.stderr.write(f'Error running fusion: {e}')
