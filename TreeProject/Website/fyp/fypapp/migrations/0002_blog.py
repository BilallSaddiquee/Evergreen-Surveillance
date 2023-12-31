# Generated by Django 4.1.2 on 2022-12-22 14:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fypapp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Blog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('author_name', models.CharField(max_length=30)),
                ('title', models.CharField(max_length=200)),
                ('tags', models.CharField(max_length=300)),
                ('slug', models.SlugField(max_length=150)),
                ('photo', models.ImageField(upload_to='media')),
                ('blog', models.TextField()),
                ('published_date', models.DateTimeField(blank=True)),
            ],
        ),
    ]
