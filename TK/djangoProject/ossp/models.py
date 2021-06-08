from django.db import models

# GameRecords
class GameRecords(models.Model):
    id = models.BigAutoField(primary_key=True)
    game = models.ForeignKey('Games', models.DO_NOTHING)
    user = models.ForeignKey('Users', models.DO_NOTHING)
    x = models.IntegerField(blank=True, null=True)
    y = models.IntegerField(blank=True, null=True)
    is_finish = models.IntegerField(blank=True, null=True)
    stone_status = models.IntegerField(blank=True, null=True)
    unallowed_list = models.CharField(max_length=255, blank=True, null=True)
    review_list = models.CharField(max_length=255, blank=True, null=True)
    created_datetime = models.DateTimeField(blank=True, null=True)


    class Meta:
        managed = False
        db_table = 'game_records'


class Games(models.Model):
    id = models.BigAutoField(primary_key=True)
    title = models.CharField(max_length=45, blank=True, null=True)
    game_mode = models.IntegerField(blank=True, null=True)
    participanta = models.IntegerField(null=False)
    participantb = models.IntegerField(null=False)
    game_status = models.IntegerField(blank=True, null=True)
    thumbnail_img_dir = models.CharField(max_length=50, blank=True, null=True)
    img_dir_list = models.CharField(max_length=225, blank=True, null=True)
    created_datetime = models.DateTimeField(blank=True, null=True)
    updated_datetime = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'games'


class Roles(models.Model):
    id = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'roles'


class UserRoles(models.Model):
    user = models.ForeignKey('Users', models.DO_NOTHING)
    role = models.ForeignKey(Roles, models.DO_NOTHING)

    class Meta:
        managed = False
        db_table = 'user_roles'


class Users(models.Model):
    id = models.BigAutoField(primary_key=True)
    nickname = models.CharField(max_length=45)
    level = models.IntegerField(blank=True, null=True)
    password = models.CharField(max_length=60)
    profile_img_dir = models.CharField(max_length=255, blank=True, null=True)
    created_datetime = models.DateTimeField(blank=True, null=True)
    updated_datetime = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'users'

class Test(models.Model):
    name = models.CharField(max_length=30, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'test'