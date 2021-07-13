from django.db import models


class ShgtDf2021(models.Model):
    code = models.TextField(primary_key=True)
    name = models.TextField(blank=True, null=True)
    todaycloseprice = models.TextField(db_column='todayClosePrice', blank=True, null=True)  # Field name made lowercase.
    todayup = models.TextField(db_column='todayUp', blank=True, null=True)  # Field name made lowercase.
    todayquantity = models.TextField(db_column='todayQuantity', blank=True, null=True)  # Field name made lowercase.
    todayvalue = models.TextField(db_column='todayValue', blank=True, null=True)  # Field name made lowercase.
    circulaterate = models.TextField(db_column='circulateRate', blank=True, null=True)  # Field name made lowercase.
    totalrate = models.TextField(db_column='totalRate', blank=True, null=True)  # Field name made lowercase.
    addnumber = models.TextField(db_column='addNumber', blank=True, null=True)  # Field name made lowercase.
    addvalue = models.TextField(db_column='addValue', blank=True, null=True)  # Field name made lowercase.
    addvaluerate = models.TextField(db_column='addValueRate', blank=True, null=True)  # Field name made lowercase.
    addvalueratecirculate = models.TextField(db_column='addValueRateCirculate', blank=True, null=True)  # Field name made lowercase.
    addvalueratetotal = models.TextField(db_column='addValueRateTotal', blank=True, null=True)  # Field name made lowercase.
    trade = models.TextField(blank=True, null=True)
    date = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'shgt_df_2021'


# 研究报告表
class MResearchReport(models.Model):
    code = models.TextField(primary_key=True)
    name = models.TextField(blank=True, null=True)
    date = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'm_research_report'
