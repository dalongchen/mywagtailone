<template>
  <el-container>
    <el-aside width="65px">
      <el-row
        @click="stock_detail(ite)"
        type="flex"
        class="row-bg"
        justify="start"
        v-for="ite in lis"
        :key="ite.id"
      >{{ ite }}</el-row>
    </el-aside>
    <el-main>
      <el-row><ElTableSon :son_props="d.sina"/>{{ d.sina.f }}</el-row>
      <el-row><ElTableSon :son_props="d.finance"/>{{ d.finance.f }}</el-row>
      <el-row><ElTableSon :son_props="d.st_ach"/>{{ d.st_ach.f }}</el-row>
      <el-row><ElTableSon :son_props="d.per_for"/>{{ d.per_for.f }}</el-row>
      <el-row><ElTableSon :son_props="d.ins_pos.ten_share"/>{{ d.ins_pos.ten_share.f }} </el-row>
      <el-row><ElTableSon :son_props="d.ins_pos.ten_current_share"/>{{ d.ins_pos.ten_current_share.f }} </el-row>
      <el-row><ElTableSon :son_props="d.ins_pos.institution_position_son"/>{{ d.ins_pos.institution_position_son.f }} </el-row>
      <el-row><ElTableSon :son_props="d.ins_pos.lift_ban"/>{{ d.ins_pos.lift_ban.f }} </el-row>
      <el-row><ElTableSon :son_props="d.lgt"/>{{ d.lgt.f }} </el-row>
      <el-row><ElTableSon :son_props="d.add_subtract"/>{{ d.add_subtract.f }} </el-row>
      <el-row><ElTableSon :son_props="d.manager_a"/>{{ d.manager_a.f }} </el-row>
      <el-row><ElTableSon :son_props="d.rz"/>{{ d.rz.f }} </el-row>
      <el-row><ElTableSon :son_props="d.capital_inflow"/>{{ d.capital_inflow.f }} </el-row>
      <el-row><ElTableSon :son_props="d.institution_res"/>{{ d.institution_res.f }} </el-row>
      <el-row><ElTableSon :son_props="d.share_num"/>{{ d.share_num.f }} </el-row>
      <el-row><ElTableSon :son_props="d.dragon_tiger"/>{{ d.dragon_tiger.f }} </el-row>
      <el-row><ElTableSon :son_props="d.ins_re_re"/>{{ d.ins_re_re.f }} </el-row>
      <el-row><ElTableSon :son_props="d.xq_discuss"/>{{ d.xq_discuss.f }} </el-row>
      <el-row><ElTableSon :son_props="d.xq_new"/>{{ d.xq_new.f }} </el-row>
      <el-row><ElTableSon :son_props="d.bai"/>{{ d.bai.f }} </el-row>
      <el-row><ElTableSon :son_props="d.st_notice"/>{{ d.st_notice.f }} </el-row>
    </el-main>
  </el-container>
</template>

<script lang="ts">
import { defineComponent, ref, reactive } from "vue";
import ElTableSon from "@/components/ElTableSon.vue";
import axios from "axios";

export default defineComponent({
  name: "Tail",
  components: {
    ElTableSon,
  },
  setup() {
    const d = reactive({
      sina: {},
      finance: {},
      st_ach: {},
      per_for: {},
      ins_pos: {
        ten_share:{},
        ten_current_share:{},
        institution_position_son:{},
        lift_ban:{},
      },
      lgt:{},
      add_subtract:{},
      manager_a:{},
      rz:{},
      capital_inflow:{},
      institution_res:{},
      share_num:{},
      dragon_tiger:{},
      ins_re_re:{},
      xq_discuss:{},
      xq_new:{},
      bai:{},
      st_notice:{},
    });
    const lis = ref();
    axios
      .get("http://127.0.0.1:8000/datatables/stock_details/")
      .then(response => {
        lis.value = response.data;
      })
      .catch(error => {
        lis.value = [error];
      });
    function stock_detail(st: string) {
      console.log(st);
      axios.get("http://127.0.0.1:8000/datatables/stock_details/", {
          params: { st: st }
        }).then(response => {
          let r = response.data;
          // console.log(r.st_ach)
          /*                */
          if (r.sina == "" || r.sina == null ) {
            d.sina = {
              f: "新浪行情为空",
            }  
          }else{
            d.sina = {
              width: 120,
              caption: "新浪实时",
              flag: "2",
              data: r.sina,
              th: [
                {prop: "0", propName: "日期"},
                {prop: "1", propName: "时间"},
                {prop: "2", propName: "名称"},
                {prop: "3", propName: "现价"},
                {prop: "4", propName: "成交量"},
                {prop: "5", propName: "成交额"},
              ]
            }
          } 
          if (r.finance == "" || r.finance == null) {
            d.finance = {
                f: "财务数据为空",
            };
          }else{
            d.finance = {
              caption: "财务",
              flag: "3",
              data: r.finance,
              th: [                
                {prop: "0", width: 125},
                {prop: "1", width: 103},
                {prop: "2", width: 103},
                {prop: "3", width: 105},
                {prop: "4", width: 105},
                {prop: "5", width: 105},
                {prop: "6", width: 105},
                {prop: "7", width: 105},
                {prop: "8", width: 105},
                {prop: "9", width: 105},
                
                {prop: "10", width: 105},
                {prop: "11", width: 105},
                {prop: "12", width: 105},
                {prop: "13", width: 105},
                {prop: "14", width: 105},
                {prop: "15", width: 105},
                {prop: "16", width: 105},
                {prop: "17", width: 105},
                {prop: "18", width: 105},
                {prop: "19", width: 105},
                
                {prop: "20", width: 105},
                {prop: "21", width: 105},
                {prop: "22", width: 105},
                {prop: "23", width: 105},
                {prop: "24", width: 105},
                {prop: "25", width: 105},
                {prop: "26", width: 105},
                {prop: "27", width: 105},
                {prop: "28", width: 105},
                {prop: "29", width: 105},
                
                {prop: "30", width: 105},
                {prop: "31", width: 105},
                {prop: "32", width: 105},
                {prop: "33", width: 105},
                {prop: "34", width: 105},
              ]
            };
          } 
          if (r.st_ach == "" || r.st_ach == null) {
            d.st_ach = {
                f: "业绩数据为空",
            };
          }else{
            // console.log(r.st_ach)
            d.st_ach = {
              caption: "业绩",
              flag: "3",
              data: r.st_ach,  
              th: [                
                {prop: "0", width: 70},
                {prop: "1", width: 105},
                {prop: "2", width: 105},
                {prop: "3", width: 105},
                {prop: "4", width: 105},
                {prop: "5", width: 105},
                {prop: "6", width: 105},
                {prop: "7", width: 105},
                {prop: "8", width: 105},
                {prop: "9", width: 105},
                {prop: "10", width: 105},
                {prop: "11", width: 105},
                {prop: "12", width: 105},
                {prop: "13", width: 105},
                {prop: "14", width: 105},
              ]
            };
          } 
          if (r.per_for == "" || r.per_for == null) {
            d.per_for = {
                f: "业绩预告为空",
            };
          }else{
            // console.log(r.st_ach)
            d.per_for = {
              caption: "业绩预告",
              flag: "3",
              data: r.per_for,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 105},
                {prop: "2", width: 105},
                {prop: "3", width: 105},
                {prop: "4", width: 105},
                {prop: "5", width: 105},
                {prop: "6", width: 205, t: true},
                {prop: "7", width: 305, t: true},
              ]
            };
          } 
          if (r.ins_pos.ten_share == "" || r.ins_pos.ten_share == null) {
            d.ins_pos.ten_share = {
                f: "十大股东为空",
            };
          }else{
            // console.log(r.ins_pos.ten_share)
            d.ins_pos.ten_share = {
              caption: "十大股东",
              flag: "4",
              data: r.ins_pos.ten_share,  
              th: [                
                {prop: "0", width: 300},
                {prop: "1", width: 95},
                {prop: "2", width: 100},
                {prop: "3", width: 100},
                {prop: "4", width: 100},
                {prop: "5", width: 100},
              ]
            };
          }
          if (r.ins_pos.ten_current_share == "" || r.ins_pos.ten_current_share == null) {
            d.ins_pos.ten_current_share = {
                f: "十大流通股东为空",
            };
          }else{
            // console.log(r.ins_pos.ten_current_share)
            d.ins_pos.ten_current_share = {
              caption: "十大流通股东",
              flag: "4",
              data: r.ins_pos.ten_current_share,  
              th: [                
                {prop: "0", width: 300},
                {prop: "1", width: 95},
                {prop: "2", width: 100},
                {prop: "3", width: 100},
                {prop: "4", width: 100},
                {prop: "5", width: 100},
                {prop: "6", width: 100},
              ]
            };
          }
          if (r.ins_pos.institution_position_son == "" || r.ins_pos.institution_position_son == null) {
            d.ins_pos.institution_position_son = {
                f: "机构持仓为空",
            };
          }else{
            // console.log(r.ins_pos.institution_position_son)
            d.ins_pos.institution_position_son = {
              caption: "机构持仓",
              flag: "4",
              data: r.ins_pos.institution_position_son,  
              th: [                
                {prop: "0", width: 100},
                {prop: "1", width: 95},
                {prop: "2", width: 100},
                {prop: "3", width: 100},
                {prop: "4", width: 100},
              ]
            };
          }
          if (r.ins_pos.lift_ban == "" || r.ins_pos.lift_ban == null) {
            d.ins_pos.lift_ban = {
                f: "解禁为空",
            };
          }else{
            // console.log(r.st_ach)
            d.ins_pos.lift_ban = {
              caption: "解禁",
              flag: "3",
              data: r.ins_pos.lift_ban,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 105},
                {prop: "2", width: 105},
                {prop: "3", width: 105},
                {prop: "4", width: 105},
              ]
            };
          } 
          if (r.lgt == "" || r.lgt == null) {
            d.lgt = {
                f: "陆股通为空",
            };
          }else{
            // console.log(r.st_ach)
            d.lgt = {
              caption: "陆股通",
              flag: "3",
              data: r.lgt,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 105},
                {prop: "2", width: 105},
              ]
            };
          }
          if (r.add_subtract == "" || r.add_subtract == null) {
            d.add_subtract = {
                f: "股东增减持为空",
            };
          }else{
            // console.log(r.add_subtract)
            d.add_subtract = {
              caption: "股东增减持",
              flag: "3",
              data: r.add_subtract,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 80},
                {prop: "2", width: 90},
                {prop: "3", width: 80},
                {prop: "4", width: 80},
                {prop: "5", width: 80},
                {prop: "6", width: 80},
                {prop: "7", width: 105},
                {prop: "8", width: 105},
                {prop: "9", width: 105},
              ]
            };
          }
          if (r.manager_a == "" || r.manager_a == null) {
            d.manager_a = {
                f: "高管增减持为空",
            };
          }else{
            // console.log(r.add_subtract)
            d.manager_a = {
              caption: "高管增减持",
              flag: "3",
              data: r.manager_a,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 80},
                {prop: "2", width: 90},
                {prop: "3", width: 80},
                {prop: "4", width: 80},
                {prop: "5", width: 80},
                {prop: "6", width: 80},
                {prop: "7", width: 105},
                {prop: "8", width: 105},
                {prop: "9", width: 105},
              ]
            };
          }
          if (r.rz == "" || r.rz == null) {
            d.rz = {
                f: "融资融券为空",
            };
          }else{
            // console.log(r.add_subtract)
            d.rz = {
              caption: "融资融券",
              flag: "3",
              data: r.rz,  
              th: [                
                {prop: "0", width: 103},
                {prop: "1", width: 100},
                {prop: "2", width: 100},
                {prop: "3", width: 100},
                {prop: "4", width: 100},
                {prop: "5", width: 100},
              ]
            };
          }
          if (r.capital_inflow == "" || r.capital_inflow == null) {
            d.capital_inflow = {
                f: "资金买入为空",
            };
          }else{
            // console.log(r.add_subtract)
            d.capital_inflow = {
              caption: "资金买入",
              flag: "3",
              data: r.capital_inflow,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 100},
              ]
            };
          }
          if (r.institution_res == "" || r.institution_res == null) {
            d.institution_res = {
                f: "机构调研为空（90天）",
            };
          }else{
            // console.log(r.add_subtract)
            d.institution_res = {
              caption: "机构调研（90天）",
              flag: "3",
              data: r.institution_res,  
              th: [                
                {prop: "0", width: 103},
                {prop: "1", width: 105},
                {prop: "2", width: 100},
                {prop: "3", width: 120},
              ]
            };
          }
          if (r.share_num == "" || r.share_num == null) {
            d.share_num = {
                f: "股东人数为空",
            };
          }else{
            // console.log(r.add_subtract)
            d.share_num = {
              caption: "股东人数",
              flag: "3",
              data: r.share_num,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 105},
                {prop: "2", width: 100},
                {prop: "3", width: 100},
                {prop: "4", width: 100},
                {prop: "5", width: 100},
                {prop: "6", width: 120},
              ]
            };
          }
          if (r.dragon_tiger == "" || r.dragon_tiger == null) {
            d.dragon_tiger = {
                f: "龙虎榜为空",
            };
          }else{
            // console.log(r.add_subtract)
            d.dragon_tiger = {
              caption: "龙虎榜",
              flag: "5",
              data: r.dragon_tiger,  
              th: [                
                {prop: "0", width: 50},
                {prop: "1", width: 220},
                {prop: "2", width: 65},
                {prop: "3", width: 80},
                {prop: "4", width: 80},
                {prop: "5", width: 80},
                {prop: "6", width: 80},
                {prop: "7", width: 80},
                {prop: "8", width: 80},
              ]
            };
          }
          if (r.ins_re_re == "" || r.ins_re_re == null) {
            d.ins_re_re = {
                f: "研究报告空（半年）",
            };
          }else{
            // console.log(r.add_subtract)
            d.ins_re_re = {
              caption: "研究报告（半年）",
              flag: "3",
              data: r.ins_re_re,  
              th: [                
                {prop: "0", width: 105},
                {prop: "1", width: 550},
              ]
            };
          }
          if (r.xq_discuss == "" || r.xq_discuss == null) {
            d.xq_discuss = {
                f: "雪球讨论空",
            };
          }else{
            // console.log(r.add_subtract)
            d.xq_discuss = {
              caption: "雪球讨论",
              flag: "3",
              data: r.xq_discuss,  
              th: [                
                {prop: "0", width: 70},
                {prop: "1", width: 170},
                {prop: "2", width: 950},
              ]
            };
          }
          if (r.xq_new == "" || r.xq_new == null) {
            d.xq_new = {
                f: "雪球资信空",
            };
          }else{
            // console.log(r.add_subtract)
            d.xq_new = {
              caption: "雪球资信",
              flag: "3",
              data: r.xq_new,  
              th: [                
                {prop: "0", width: 70},
                {prop: "1", width: 250},
                {prop: "2", width: 950},
              ]
            };
          }
          if (r.bai == "" || r.bai == null) {
            d.bai = {
                f: "百度负面空",
            };
          }else{
            // console.log(r.add_subtract)
            d.bai = {
              caption: "百度负面",
              flag: "30",
              data: r.bai,  
              th: [                
                {prop: "0", width: 80},
                {prop: "1", width: 70},
                {prop: "2", width: 250},
                {prop: "3", width: 750},
                {prop: "4", width: 60, t: true},
              ]
            };
          }
           if (r.st_notice == "" || r.st_notice == null) {
            d.st_notice = {
                f: "个股公告空",
            };
          }else{
            // console.log(r.add_subtract)
            d.st_notice = {
              caption: "个股公告",
              flag: "3",
              data: r.st_notice,  
              th: [                
                {prop: "0", width: 200},
                {prop: "1", width: 210},
                {prop: "2", width: 700},
              ]
            };
          }
        }).catch(error => {
          console.log(error);
        });
    }
    return { lis, stock_detail, d };
  }
});
</script>
<style lang="scss">
.el-row {
    margin-bottom: 8px;
    &:last-child {
      margin-bottom: 0;
    }
  }
</style>
