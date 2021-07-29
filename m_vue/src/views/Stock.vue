<template>
  <el-container>
    <el-header>
      <el-row  class="row-bg" >
        <el-col :span="2">
            <el-button round @click="research_report()">研告和机调数量(1年)</el-button>
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
  <div>   
  </div>
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
    research_report() {
      axios
        .get("http://127.0.0.1:8000/datatables/east_money_lgt/", {
          params: { research_report: "research_report", }
        })
        .then(response => {
          console.log(response.data);
          // let r = response.data;
          // this.parent_data = { flag: "1", data: r.dc_notice_go}
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