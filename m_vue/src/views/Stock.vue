<template>
  <el-container>
    <el-header style="height: 170px; border: 1px solid #eee">
      <el-row  class="row-bg" >
        <el-col :span="4">
            <el-button round @click="choice_stock('research_report')" :disabled="f">
              {{research_report}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('east_lgt_number')" :disabled="f">
              {{east_lgt_number}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('east_finance_number')" :disabled="f">
              {{east_finance_number}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('combine')" :disabled="f">
              {{combine}}
            </el-button>
        </el-col>        
        <el-col :span="4">
            <el-button round @click="choice_stock('ths_fund_inflow0')" :disabled="f">
              {{ths_fund_inflow0}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('east_dragon')" :disabled="f">
              {{east_dragon}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('open_dragon')" :disabled="ff">
              {{open_dragon}}
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('east_lgt')" :disabled="f">
              {{east_lgt}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('ths_lgt')" :disabled="f">
              {{ths_lgt}}
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('east_finance_sh')" :disabled="f">
              {{east_finance_sh}}
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('east_finance_sz')" :disabled="f">
              {{east_finance_sz}}
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('shown_choice')" :disabled="f">
              {{shown_choice}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('pre_paid')"  :disabled="f">
              {{pre_paid}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="send_text('ths_lgt02')" :disabled="f">
              {{ths_lgt02}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('backup')" :disabled="f">
              {{backup}}
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('kzz')" :disabled="f">
              {{kzz}}
            </el-button>
        </el-col>
      </el-row>
      <el-row  class="row-bg" >
        <el-col :span="3">
            <DateTimeSon @func="getMsgSon"/>
        </el-col>
        <el-col :span="3">
            <el-button round @click="get_date" :disabled="f">{{good_notice}}</el-button>
        </el-col>
        <el-col :span="14">
            <el-input clearable placeholder="请输入内容" v-model="textarea"></el-input>
        </el-col>
        <el-col :span="2">
            <el-button round @click="send_text('ths_in')" :disabled="f">{{ number }}</el-button>
        </el-col>
      </el-row>
    </el-header>
    <el-main>
      <ElTableSon :son_props="parent_data"/>
    </el-main>
  </el-container>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import ElTableSon from "@/components/ElTableSon.vue";
import DateTimeSon from "@/components/DateTimeSon.vue";
import axios from "axios";

export default defineComponent({
  name: "Stock",
  components: {
    ElTableSon,
    DateTimeSon,
  },
  data() {
    return {
      f:false,
      ff:false,
      parent_data:{},
      msgSon:"",
      textarea:"",
      number:"提交",
      research_report:"研告和机调数量(1年)",
      east_lgt_number:"陆股通数量",
      east_finance_number:"融资融券数量",
      combine:"并集和交集",
      ths_fund_inflow0:"流入大10和大5000",
      east_dragon:"龙虎榜",
      open_dragon:"打开龙虎榜",
      east_lgt:"东财陆股通",
      ths_lgt:"同花顺陆股通",
      east_finance_sh:"上海两融",
      east_finance_sz:"深圳两融",
      shown_choice:"写入雪球和自选",
      pre_paid:"5:30后预买",
      ths_lgt02:"陆股通大于0.2",
      backup:"备份",
      kzz:"east可转债",
      good_notice:"公告利好",
    };
  },
  methods: {
    get_date() {
      this.f = true;
      this.good_notice = "工作中约30s"
      if (this.msgSon != ""){
        if (this.msgSon != null){
          //console.log(this.msgSon.toString());
          this.msgSon = this.msgSon.toString()
        }
      }
      axios.get("http://127.0.0.1:8000/datatables/east_data/", {
        params: { dc_notice: "dc_notice", d_t: this.msgSon}
      }).then(response => {
        this.parent_data = { flag: "1", data: response.data.dc_notice_go}
        //console.log(r.dc_notice_go);
        this.f = false;
        this.good_notice = "ok"
      }).catch(error => {
        console.log(error);
        this.f = false;
        this.good_notice = "error"
      });
    },
    getMsgSon(d: string) {
       this.msgSon = d;
    },
    send_text(s:string) {
      this.f = true;
      this.ths_lgt02 = "工作中30s"
      if (s == "ths_in"){
        if (this.textarea != "" && this.textarea != null){
            // console.log(this.textarea);      
            axios.get("http://127.0.0.1:8000/datatables/east_money_lgt/", {
                params: { ths_choice: "ths_choice", ths_in: this.textarea}
            }).then(response => {
              this.number = response.data.number;
              // console.log(response.data.number);
              this.f = false;
              this.ths_lgt02 = "ok"
            }).catch(error => {
              console.log(error);
              this.f = false;
              this.ths_lgt02 = "error"
            });
        }
      }else{
        if (this.msgSon != ""){
          if (this.msgSon != null){
            //console.log(this.msgSon.toString());
            this.msgSon = this.msgSon.toString()
          }
        }
        axios.get("http://127.0.0.1:8000/datatables/east_money_lgt/", {
          params: { ths_choice: "ths_choice", d_t: this.msgSon, t: s}
        }).then(response => {
          this.ths_lgt02 = response.data.number;
          // console.log(response.data.number);
          this.f = false;
        }).catch(error => {
          console.log(error);
          this.f = false;
        });
      }
    },
    choice_stock(d:string) {  
      this.f = true;
      if (d == "research_report"){
        this.research_report = "工作中约30秒";
      }else if (d == "east_lgt_number") {
        this.east_lgt_number = "工作中约30秒";
      }else if (d == "east_finance_number") {
        this.east_finance_number = "工作中约30秒";
      }else if (d == "combine") {
        this.combine = "工作中约1秒";
      }else if (d == "ths_fund_inflow0") {
        this.ths_fund_inflow0 = "工作中约60秒";
      }else if (d == "east_dragon") {
        this.east_dragon = "工作中约20秒";
      }else if (d == "open_dragon") {
        this.open_dragon = "工作中约60秒";
        this.ff = true;
      }else if (d == "east_lgt") {
        this.east_lgt = "工作中约30秒";
      }else if (d == "ths_lgt") {
        this.ths_lgt = "工作中约30秒";
      }else if (d == "east_finance_sh") {
        this.east_finance_sh = "工作中约30秒";
      }else if (d == "east_finance_sz") {
        this.east_finance_sz = "工作中约30秒";
      }else if (d == "shown_choice") {
        this.shown_choice = "工作中约30秒";
      }else if (d == "pre_paid") {
        this.pre_paid = "工作中约60秒";
      }else if (d == "backup") {
        this.backup = "工作中约60秒";
      }else if (d == "kzz") {
        this.kzz = "工作中约60秒";
      }
      // console.log(d);
      if (this.msgSon != ""){
        if (this.msgSon != null){
          //console.log(this.msgSon.toString());
          this.msgSon = this.msgSon.toString()
        }
      }
      axios.get("http://127.0.0.1:8000/datatables/east_money_lgt/", {
        params: { s: d, d_t: this.msgSon }
      }).then(response => {
        let r =response.data.number
        // console.log(response.data.number);
        if (d == "research_report"){
          this.research_report = r;
        }else if (d == "east_lgt_number") {
          this.east_lgt_number = r;
        }else if (d == "east_finance_number") {
          this.east_finance_number = r;
        }else if (d == "combine") {
          this.combine = r;
        }else if (d == "ths_fund_inflow0") {
          this.ths_fund_inflow0 = r;
        }else if (d == "east_dragon") {
          this.east_dragon = r;
        }else if (d == "open_dragon") {
          let dd = r.pop()
          for (let code of r) {
            window.open("http://data.eastmoney.com/stock/lhb," + dd + "," + code + ".html");
            let exitTime = new Date().getTime() + 1000;
            let t = new Date().getTime();
            while (t < exitTime) { // sleep 1 秒
              t = new Date().getTime()
            }
          }
          this.ff = false;
          this.open_dragon = "打开龙虎榜"
        }else if (d == "east_lgt") {
          this.east_lgt = r;
        }else if (d == "ths_lgt") {
          this.ths_lgt = r;
        }else if (d == "east_finance_sh") {
          this.east_finance_sh = r;
        }else if (d == "east_finance_sz") {
          this.east_finance_sz = r;
        }else if (d == "shown_choice") {
          this.shown_choice = r;
        }else if (d == "pre_paid") {
          this.pre_paid = r;
        }else if (d == "backup") {
          this.backup = r;
        }else if (d == "kzz") {
          this.kzz = r;
        }
        this.f = false;
      }).catch(error => {
        let rr = "error"
        // console.log(response.data.number);
        if (d == "research_report"){
          this.research_report = rr;
        }else if (d == "east_lgt_number") {
          this.east_lgt_number = rr;
        }else if (d == "east_finance_number") {
          this.east_finance_number = rr;
        }else if (d == "combine") {
          this.combine = rr;
        }else if (d == "ths_fund_inflow0") {
          this.ths_fund_inflow0 = rr;
        }else if (d == "east_dragon") {
          this.east_dragon = rr;
        }else if (d == "open_dragon") {
          this.open_dragon = rr;
        }else if (d == "east_lgt") {
          this.east_lgt = rr;
        }else if (d == "ths_lgt") {
          this.ths_lgt = rr;
        }else if (d == "east_finance_sh") {
          this.east_finance_sh = rr;
        }else if (d == "east_finance_sz") {
          this.east_finance_sz = rr;
        }else if (d == "shown_choice") {
          this.shown_choice = rr;
        }else if (d == "pre_paid") {
          this.pre_paid = rr;
        }else if (d == "backup") {
          this.backup = rr;
        }else if (d == "kzz") {
          this.kzz = rr;
        }
        this.f = false;
        this.ff = false;
        console.log(error);
      });
    },
  },
});
</script>
<style scoped lang="scss">
 .el-row {
    margin-top: 10px;
    &:last-child {
      margin-bottom: 0;
    }
  }
</style>