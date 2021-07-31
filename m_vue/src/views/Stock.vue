<template>
  <el-container>
    <el-header style="height: 130px; border: 1px solid #eee">
      <el-row  class="row-bg" >
        <el-col :span="4">
            <el-button round @click="choice_stock('research_report')">
              <div v-if="research_report === ''">研告和机调数量(1年)</div>
              <div v-else>{{research_report}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('east_lgt_number')">
              <div v-if="east_lgt_number === ''">陆股通数量</div>
              <div v-else>{{east_lgt_number}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('east_finance_number')">
              <div v-if="east_finance_number === ''">融资融券数量</div>
              <div v-else>{{east_finance_number}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('combine')">
              <div v-if="combine === ''">并集和交集</div>
              <div v-else>{{combine}}</div>
            </el-button>
        </el-col>        
        <el-col :span="4">
            <el-button round @click="choice_stock('ths_fund_inflow0')">
              <div v-if="ths_fund_inflow0 === ''">流入大10和大2500</div>
              <div v-else>{{ths_fund_inflow0}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('east_dragon')">
              <div v-if="east_dragon === ''">龙虎榜</div>
              <div v-else>{{east_dragon}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('open_dragon')">打开龙虎榜</el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('east_lgt')">
              <div v-if="east_lgt === ''">东财陆股通</div>
              <div v-else>{{east_lgt}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('ths_lgt')">
              <div v-if="ths_lgt === ''">同花顺陆股通</div>
              <div v-else>{{ths_lgt}}</div>
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('east_finance_sh')">
              <div v-if="east_finance_sh === ''">上海两融</div>
              <div v-else>{{east_finance_sh}}</div>
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('east_finance_sz')">
              <div v-if="east_finance_sz === ''">深圳两融</div>
              <div v-else>{{east_finance_sz}}</div>
            </el-button>
        </el-col>
        <el-col :span="4">
            <el-button round @click="choice_stock('shown_choice')">
              <div v-if="shown_choice === ''">写入雪球和自选</div>
              <div v-else>{{shown_choice}}</div>
            </el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="choice_stock('pre_paid')">
              <div v-if="pre_paid === ''">5:30后预买</div>
              <div v-else>{{pre_paid}}</div>
            </el-button>
        </el-col>
      </el-row>
      <el-row  class="row-bg" >
        <el-col :span="3">
            <DateTimeSon @func="getMsgSon"/>
        </el-col>
        <el-col :span="2">
            <el-button round @click="get_date">公告利好</el-button>
        </el-col>
        <el-col :span="14">
            <el-input clearable placeholder="请输入内容" v-model="textarea"></el-input>
        </el-col>
        <el-col :span="2">
            <el-button round @click="send_text">提交 {{ number }}</el-button>
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
      parent_data:{},
      msgSon:"",
      textarea:"",
      number:"",
      research_report:"",
      east_lgt_number:"",
      east_finance_number:"",
      combine:"",
      ths_fund_inflow0:"",
      east_dragon:"",
      east_lgt:"",
      ths_lgt:"",
      east_finance_sh:"",
      east_finance_sz:"",
      shown_choice:"",
      pre_paid:"",
    };
  },
  methods: {
    get_date() {
      if (this.msgSon != ""){
        if (this.msgSon != null){
          //console.log(this.msgSon.toString());
          this.msgSon = this.msgSon.toString()
        }
      }
      axios
        .get("http://127.0.0.1:8000/datatables/east_data/", {
          params: { dc_notice: "dc_notice", d_t: this.msgSon}
        })
        .then(response => {
          this.parent_data = { flag: "1", data: response.data.dc_notice_go}
          //console.log(r.dc_notice_go);
        })
        .catch(error => {
          console.log(error);
        });
    },
    getMsgSon(d: string) {
       this.msgSon = d;
       //console.log("parent" + this.msgSon);
    },
    send_text() {
      if (this.textarea != "" && this.textarea != null){
          // console.log(this.textarea);      
          axios.get("http://127.0.0.1:8000/datatables/east_money_lgt/", {
              params: { ths_choice: "ths_choice", ths_in: this.textarea}
          }).then(response => {
            this.number = response.data.number;
            // console.log(response.data.number);
          }).catch(error => {
            console.log(error);
          });
      }
    },
    choice_stock(d:string) {
      // console.log(d);
      if (this.msgSon != ""){
        if (this.msgSon != null){
          //console.log(this.msgSon.toString());
          this.msgSon = this.msgSon.toString()
        }
      }
      axios
        .get("http://127.0.0.1:8000/datatables/east_money_lgt/", {
          params: { s: d, d_t: this.msgSon }
        })
        .then(response => {
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
            let d = r.pop()
            for (let code of r) {
              // console.log("http://data.eastmoney.com/stock/lhb," + d + "," + code + ".html");
              let exitTime = new Date().getTime() + 1500;
              let t = new Date().getTime();
              while (t < exitTime) { // sleep 1 秒
                t = new Date().getTime()
              }
              window.open("http://data.eastmoney.com/stock/lhb," + d + "," + code + ".html");
            }
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
          }
        })
        .catch(error => {
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