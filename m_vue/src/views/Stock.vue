<template>
  <el-container>
    <el-aside width="220px">
      <el-row type="flex" class="row-bg" justify="start">
        <el-col :span="7">
            <DateTimeSon @func="getMsgSon"/>
            <el-button round @click="get_date">公告利好</el-button>
        </el-col>
      </el-row>
      
    </el-aside>
    <el-main>
      <el-row type="flex" class="row-bg" justify="start">
        <el-col :span="18">
            <el-input
              clearable
              placeholder="请输入内容"
              v-model="textarea"
            >
            </el-input>
        </el-col>
        <el-button round @click="send_text">提交 {{ number }}</el-button>
      </el-row>
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
          let r = response.data;
          this.parent_data = { flag: "1", data: r.dc_notice_go}
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
  }
  
});
</script>
<style scoped lang="scss">
</style>