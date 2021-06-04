# Generated by Django 3.2.4 on 2021-06-04 10:43

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GameRecords',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('x', models.IntegerField(blank=True, null=True)),
                ('y', models.IntegerField(blank=True, null=True)),
                ('is_finish', models.IntegerField(blank=True, null=True)),
                ('stone_status', models.IntegerField(blank=True, null=True)),
                ('created_datetime', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'db_table': 'game_records',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Games',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('title', models.CharField(blank=True, max_length=45, null=True)),
                ('game_mode', models.IntegerField(blank=True, null=True)),
                ('participanta', models.IntegerField()),
                ('participantb', models.IntegerField()),
                ('game_status', models.IntegerField(blank=True, null=True)),
                ('thumbnail_img_dir', models.CharField(blank=True, max_length=50, null=True)),
                ('img_dir_list', models.CharField(blank=True, max_length=225, null=True)),
                ('created_datetime', models.DateTimeField(blank=True, null=True)),
                ('updated_datetime', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'db_table': 'games',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Roles',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(blank=True, max_length=45, null=True)),
            ],
            options={
                'db_table': 'roles',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='UserRoles',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
            ],
            options={
                'db_table': 'user_roles',
                'managed': False,
            },
        ),
        migrations.CreateModel(
            name='Users',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('nickname', models.CharField(max_length=45)),
                ('level', models.IntegerField(blank=True, null=True)),
                ('password', models.CharField(max_length=60)),
                ('profile_img_dir', models.CharField(blank=True, max_length=255, null=True)),
                ('created_datetime', models.DateTimeField(blank=True, null=True)),
                ('updated_datetime', models.DateTimeField(blank=True, null=True)),
            ],
            options={
                'db_table': 'users',
                'managed': False,
            },
        ),
    ]
