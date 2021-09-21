<template>
<el-container>
    <el-header style="height: 70px; border: 1px solid #eee">
      <el-row  class="row-bg" >
        <el-col :span="3">
            <el-button round @click="get_date" :disabled="f">{{pre_paid}}</el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="show_pre_paid" :disabled="f">{{show_pre}}</el-button>
        </el-col>
        <el-col :span="3">
            <el-button round @click="backup_close_buy" :disabled="f">{{backup_close}}</el-button>
        </el-col>
      </el-row>
    </el-header>
    <el-main>
        <TableTrade :son_props="parent_data"/>
    </el-main>
  </el-container>
</template>

<script lang="ts">
import { defineComponent } from "vue";
import TableTrade from "@/components/TableTrade.vue";
import axios from "axios";

export default defineComponent ({
  name: 'About',
  components: {
    TableTrade,
    // DateTimeSon,
  },
  data() {
    return {
      f:false,
      parent_data:{}, 
      pre_paid:"预埋列表",  
      show_pre:"显示预买表",  
      backup_close:"备份收盘价",  
    };
  },
  methods: {
     get_date() {
      this.f = true;
      this.pre_paid = "工作中约30s"
      axios.get("http://127.0.0.1:8000/datatables/easy_trade/", {
        params: { s: "pre_paid"}
      }).then(response => {
        // this.parent_data = { flag: "1", data: response.data.pre_paid}
        console.log(response.data.pre_paid);
        this.f = false;
        this.pre_paid = "预买成功"
      }).catch(error => {
        console.log(error);
        this.f = false;
        this.pre_paid = "error"
      });
    },

    show_pre_paid() {
      this.f = true;
      this.show_pre = "工作中约30s"
      axios.get("http://127.0.0.1:8000/datatables/easy_trade/", {
        params: { s: "show_pre"}
      }).then(response => {
        this.parent_data = { flag: "1", data: response.data.show_pre}
        // console.log(response.data);
        this.f = false;
        this.show_pre = "显示预买"
      }).catch(error => {
        console.log(error);
        this.f = false;
        this.show_pre = "error"
      });
    },

    backup_close_buy() {
      this.f = true;
      this.backup_close = "工作中约30s"
      axios.get("http://127.0.0.1:8000/datatables/easy_trade/", {
        params: { s: "backup_close"}
      }).then(response => {
        console.log(response.data);
        this.f = false;
        this.backup_close = "ok"
      }).catch(error => {
        console.log(error);
        this.f = false;
        this.backup_close = "error"
      });
    },



  },
  
})

// npm install vue-class-component vue-property-decorator --save-dev
// npm uninstall vue-property-decorator --save-dev
</script>
<style scoped lang="scss">
 .el-row {
    margin-top: 10px;
    &:last-child {
      margin-bottom: 0;
    }
  }
</style>
